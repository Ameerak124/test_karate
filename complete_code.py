import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance, chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from yake import KeywordExtractor
import hdbscan
import umap.umap_ as umap

class OilGasDriftDetector:
    def __init__(self, reference_texts, num_key_phrases=30):
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
        # Automatically discover phrases and categories
        self.key_phrases, self.category_map = self._discover_phrases_categories(
            reference_texts, 
            num_key_phrases
        )
        
        # Process reference data
        self.reference_features = self._process_batch(reference_texts)
        self._fit_scalers()
    
    def _discover_phrases_categories(self, texts, num_phrases):
        """Full automated discovery of phrases and categories"""
        # Extract candidate phrases
        candidates = self._extract_candidates(texts, num_phrases)
        
        # Cluster phrases into semantic categories
        return self._cluster_phrases(candidates)
    
    def _extract_candidates(self, texts, num_phrases):
        """Combine YAKE and TF-IDF for phrase extraction"""
        # YAKE extraction
        kw_extractor = KeywordExtractor(
            lan="en", n=3, dedupLim=0.8, top=num_phrases*2
        )
        yake_phrases = set()
        for text in texts:
            keywords = kw_extractor.extract_keywords(text)
            yake_phrases.update([kw[0].lower() for kw in keywords])
        
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
        tfidf.fit(texts)
        tfidf_phrases = set(tfidf.get_feature_names_out())
        
        return list(yake_phrases.union(tfidf_phrases))
    
    def _cluster_phrases(self, phrases):
        """Semantic clustering of phrases into categories"""
        # Get phrase embeddings
        phrase_embeddings = self.embedder.encode(phrases)
        
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=5, random_state=42)
        reduced_embeds = reducer.fit_transform(phrase_embeddings)
        
        # Cluster phrases
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            cluster_selection_epsilon=0.5,
            min_samples=1
        )
        clusters = clusterer.fit_predict(reduced_embeds)
        
        # Organize phrases by cluster
        category_phrases = {}
        for phrase, cluster_id in zip(phrases, clusters):
            if cluster_id == -1:
                continue  # Ignore noise
            if cluster_id not in category_phrases:
                category_phrases[cluster_id] = []
            category_phrases[cluster_id].append(phrase)
        
        # Name categories using central phrases
        final_categories = {}
        for cid, phrases in category_phrases.items():
            if len(phrases) < 3:
                continue
            
            # Find most representative phrase
            embeddings = self.embedder.encode(phrases)
            centroid = np.mean(embeddings, axis=0)
            sims = [1 - cosine(e, centroid) for e in embeddings]
            cat_name = phrases[np.argmax(sims)]
            
            final_categories[cat_name] = phrases
        
        # Create reverse mapping for detection
        phrase_to_category = {}
        for cat, phrases in final_categories.items():
            for phrase in phrases:
                phrase_to_category[phrase] = cat
                
        return final_categories.keys(), phrase_to_category

    def _process_text(self, text):
        """Clean OCR text"""
        text = text.lower().replace('\n', ' ')
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def _extract_features(self, text):
        """Create comprehensive feature vector"""
        # Semantic embedding
        embedding = self.embedder.encode(text, convert_to_tensor=True).cpu().numpy()
        
        # Phrase category distribution
        category_counts = {cat:0 for cat in self.key_phrases}
        words = text.split()
        for phrase, cat in self.category_map.items():
            if phrase in text:
                category_counts[cat] += 1
        
        # Text statistics
        stats = [
            len(words),
            len(set(words))/len(words) if len(words) > 0 else 0,
            sum(1 for w in words if len(w) > 7)/len(words) if len(words) > 0 else 0
        ]
        
        return np.concatenate([
            embedding,
            list(category_counts.values()),
            stats
        ])
    
    def _process_batch(self, texts):
        return np.array([self._extract_features(self._process_text(t)) for t in texts])
    
    def _fit_scalers(self):
        self.scaler.fit(self.reference_features)
        scaled = self.scaler.transform(self.reference_features)
        self.pca.fit(scaled)
    
    def detect_drift(self, new_texts, alpha=0.05):
        """Comprehensive drift detection"""
        new_features = self._process_batch(new_texts)
        
        # Dimensionality reduction
        ref_scaled = self.scaler.transform(self.reference_features)
        new_scaled = self.scaler.transform(new_features)
        
        ref_pca = self.pca.transform(ref_scaled)
        new_pca = self.pca.transform(new_scaled)
        
        # 1. Semantic drift (Wasserstein)
        sem_drift = np.mean([
            wasserstein_distance(ref_pca[:,i], new_pca[:,i])
            for i in range(ref_pca.shape[1])
        ])
        
        # 2. Category distribution (Chi-square)
        cat_drift = {}
        for cat in self.key_phrases:
            ref_counts = [t.count(cat) for t in self.reference_texts]
            new_counts = [t.count(cat) for t in new_texts]
            _, pval = chi2_contingency([
                np.histogram(ref_counts, bins=5)[0],
                np.histogram(new_counts, bins=5)[0]
            ])
            cat_drift[cat] = pval < alpha
        
        # 3. Statistical drift (KS test)
        stats_drift = []
        for i in range(ref_pca.shape[1]):
            _, pval = ks_2samp(ref_pca[:,i], new_pca[:,i])
            stats_drift.append(pval < alpha)
        
        return {
            'semantic_drift': sem_drift,
            'category_drift': cat_drift,
            'stats_drift': np.mean(stats_drift),
            'overall_alert': (
                sem_drift > 0.3 or 
                any(cat_drift.values()) or 
                np.mean(stats_drift) > 0.5
            )
        }

# Example Usage
if __name__ == "__main__":
    # Sample training documents (OCR text)
    reference_docs = [
        "Oil and Gas Lease Agreement effective 2023-01-01... Primary term 36 months "
        "with royalty interest of 18.75%. Depth limitation clause restricts drilling "
        "below 10,000 feet. Surface rights reserved to landowner.",
        
        "Memorandum of Lease for County records... Pooling provisions allow for "
        "160-acre units. Delay rental payments set at $5/acre annually. "
        "Habendum clause maintains lease while operations continue.",
    ]
    
    # Initialize detector
    detector = OilGasDriftDetector(reference_docs)
    
    print("Discovered Categories:", detector.key_phrases)
    
    # Test documents
    test_docs = [
        "Surface Use Agreement for access roads construction... No mineral rights included.",
        "Amended Royalty Clause reducing payments to 15% with transportation deductions.",
        "Clinical Trial Agreement between PharmaCorp and Hospital for Drug XYZ-123."
    ]
    
    # Detect drift
    results = detector.detect_drift(test_docs)
    
    print("\nDrift Results:")
    print(f"Semantic Drift Score: {results['semantic_drift']:.3f}")
    print("Category Drifts:")
    for cat, drifted in results['category_drift'].items():
        print(f" - {cat}: {'DRIFT' if drifted else 'OK'}")
    print(f"Overall Alert: {results['overall_alert']}")
