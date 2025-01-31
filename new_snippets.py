import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance, chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from yake import KeywordExtractor
from collections import defaultdict

class LeaseDriftDetector:
    def __init__(self, reference_texts, num_phrases=15, extraction_method='yake'):
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
        # Automatically discover key phrases
        self.key_phrases = self._discover_key_phrases(
            reference_texts, 
            num_phrases,
            extraction_method
        )
        
        # Process reference data
        self.reference_texts = reference_texts
        self.reference_features = self._process_batch(reference_texts)
        self._fit_scalers()
    
    def _discover_key_phrases(self, texts, num_phrases, method='yake'):
        """Automatically extract and categorize key phrases"""
        # Extract candidate phrases
        if method == 'yake':
            kw_extractor = KeywordExtractor(
                lan="en", n=3, dedupLim=0.8, top=num_phrases*3
            )
            candidates = []
            for text in texts:
                keywords = kw_extractor.extract_keywords(text)
                candidates.extend([kw[0].lower() for kw in keywords])
        else:  # TF-IDF fallback
            vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
            vectorizer.fit(texts)
            candidates = vectorizer.get_feature_names_out()
        
        # Semantic categorization
        category_seeds = {
            'lease_structure': ['effective date', 'primary term', 'habendum clause'],
            'financial_terms': ['royalty interest', 'delay rental', 'shut-in royalty'],
            'operations': ['depth limitation', 'pooling clause', 'surface rights']
        }
        
        # Create category embeddings
        category_embeddings = {}
        for cat, seeds in category_seeds.items():
            category_embeddings[cat] = self.embedder.encode(seeds)
        
        # Categorize phrases
        phrase_categories = defaultdict(list)
        for phrase in set(candidates):
            phrase_embed = self.embedder.encode(phrase)
            similarities = {}
            for cat, seed_embs in category_embeddings.items():
                sim_scores = [1 - cosine(phrase_embed, se) for se in seed_embs]
                similarities[cat] = np.max(sim_scores)
            
            best_cat = max(similarities, key=similarities.get)
            if similarities[best_cat] > 0.55:  # Similarity threshold
                phrase_categories[best_cat].append((
                    phrase, similarities[best_cat]
                ))
        
        # Select top phrases per category
        final_phrases = {}
        for cat in category_seeds.keys():
            sorted_phrases = sorted(phrase_categories.get(cat, []), 
                                  key=lambda x: x[1], reverse=True)
            final_phrases[cat] = [p[0] for p in sorted_phrases[:num_phrases//3]]
            
        return final_phrases

    def _process_text(self, text):
        """Clean and normalize text"""
        text = text.lower().replace('\n', ' ')
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_features(self, text):
        """Create multi-dimensional feature vector"""
        # Semantic embedding
        embedding = self.embedder.encode(text, convert_to_tensor=True).cpu().numpy()
        
        # Phrase frequencies
        phrase_features = []
        for cat, phrases in self.key_phrases.items():
            phrase_features.extend([text.count(phrase) for phrase in phrases])
        
        # Text statistics
        words = text.split()
        stats_features = [
            len(words),  # Total words
            len(set(words)) / len(words) if len(words) > 0 else 0,  # Unique ratio
            sum(1 for word in words if len(word) > 7) / len(words) if len(words) > 0 else 0,  # Long words
            text.count('exhibit')  # Document structure marker
        ]
        
        return np.concatenate([embedding, phrase_features, stats_features])

    def _process_batch(self, texts):
        """Process multiple texts into feature matrix"""
        return np.array([self._extract_features(self._process_text(t)) for t in texts])

    def _fit_scalers(self):
        """Fit normalization and PCA"""
        self.scaler.fit(self.reference_features)
        scaled_ref = self.scaler.transform(self.reference_features)
        self.pca.fit(scaled_ref)

    def detect_drift(self, new_texts, alpha=0.05):
        """Detect and quantify data drift"""
        # Feature extraction
        new_features = self._process_batch(new_texts)
        
        # Dimensionality reduction
        ref_scaled = self.scaler.transform(self.reference_features)
        new_scaled = self.scaler.transform(new_features)
        
        ref_pca = self.pca.transform(ref_scaled)
        new_pca = self.pca.transform(new_scaled)
        
        # 1. Semantic drift (Wasserstein distance)
        semantic_drifts = [
            wasserstein_distance(ref_pca[:,i], new_pca[:,i]) 
            for i in range(ref_pca.shape[1])
        ]
        
        # 2. Phrase distribution drift (Chi-square test)
        phrase_drift = {}
        for cat in self.key_phrases:
            ref_counts = [sum(t.count(p) for p in self.key_phrases[cat]) 
                        for t in self.reference_texts]
            new_counts = [sum(t.count(p) for p in self.key_phrases[cat]) 
                        for t in new_texts]
            
            # Create contingency table
            bins = np.histogram_bin_edges(ref_counts + new_counts, bins='auto')
            ref_hist, _ = np.histogram(ref_counts, bins=bins)
            new_hist, _ = np.histogram(new_counts, bins=bins)
            
            _, p_value, _, _ = chi2_contingency([ref_hist, new_hist])
            phrase_drift[cat] = {
                'p_value': p_value,
                'drift_detected': p_value < alpha
            }
        
        # 3. Statistical drift (KS test)
        stats_drift = []
        for i in range(ref_pca.shape[1]):
            _, p_value = ks_2samp(ref_pca[:,i], new_pca[:,i])
            stats_drift.append(p_value < alpha)
        
        return {
            'semantic_drift_score': np.mean(semantic_drifts),
            'phrase_drift': phrase_drift,
            'stats_drift_ratio': np.mean(stats_drift),
            'overall_drift': (
                np.mean(semantic_drifts) > 0.3 or 
                any([v['drift_detected'] for v in phrase_drift.values()]) or
                np.mean(stats_drift) > 0.5
            )
        }

# Example Usage
if __name__ == "__main__":
    # Sample training documents (OCR output)
    reference_docs = [
        "Oil and Gas Lease Agreement effective January 1 2023 between... "
        "Primary term of three (3) years with royalty interest at 18.75%... "
        "Depth limitation clause restricting drilling below 10000 feet...",
        
        "Memorandum of Lease for County records... "
        "Surface rights reserved to Landowner with pooling provisions... "
        "Delay rental payments set at $5 per acre annually...",
    ]
    
    # Initialize detector
    detector = LeaseDriftDetector(reference_docs, num_phrases=12)
    
    # Test documents
    test_docs = [
        "Surface Use Agreement permitting access roads... (no lease terms)",
        "Amended Royalty Clause reducing payments to 15% with new deductions",
        "Clinical Trial Agreement unrelated to oil/gas leasing"
    ]
    
    # Detect drift
    results = detector.detect_drift(test_docs)
    
    print("Key Phrases Discovered:")
    print(detector.key_phrases)
    
    print("\nDrift Detection Results:")
    print(f"Semantic Drift Score: {results['semantic_drift_score']:.3f}")
    print(f"Phrase Drift Detection:")
    for cat, vals in results['phrase_drift'].items():
        print(f" - {cat}: {'DRIFT' if vals['drift_detected'] else 'OK'} (p={vals['p_value']:.4f})")
    print(f"Overall Drift Alert: {results['overall_drift']}")




