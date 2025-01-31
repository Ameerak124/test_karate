def _discover_key_phrases(self, texts, num_phrases):
    """Fully automated phrase and category discovery"""
    # Extract candidate phrases
    candidates = self._extract_candidate_phrases(texts, num_phrases)
    
    # Cluster phrases into semantic categories
    category_map = self._cluster_phrases(candidates)
    
    # Select top phrases per category
    final_phrases = {}
    for cat in category_map.values():
        cat_name = cat['name']
        final_phrases[cat_name] = cat['phrases'][:num_phrases//len(category_map)]
    
    return final_phrases

def _extract_candidate_phrases(self, texts, num_phrases):
    """Extract candidate phrases using combined YAKE+TF-IDF"""
    # YAKE extraction
    kw_extractor = KeywordExtractor(top=num_phrases*2)
    yake_phrases = set()
    for text in texts:
        keywords = kw_extractor.extract_keywords(text)
        yake_phrases.update([kw[0].lower() for kw in keywords])
    
    # TF-IDF extraction
    tfidf_phrases = self._extract_tfidf_phrases(texts, num_phrases)
    
    return list(yake_phrases.union(tfidf_phrases))

def _cluster_phrases(self, phrases):
    """Cluster phrases using semantic similarity"""
    embeddings = self.embedder.encode(phrases)
    
    # Dimensionality reduction
    from umap import UMAP
    reducer = UMAP(n_components=5, random_state=42)
    reduced_embeds = reducer.fit_transform(embeddings)
    
    # Density-based clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, 
                              cluster_selection_epsilon=0.5)
    clusters = clusterer.fit_predict(reduced_embeds)
    
    # Create category map
    category_map = {}
    for phrase, cluster_id in zip(phrases, clusters):
        if cluster_id not in category_map:
            category_map[cluster_id] = []
        category_map[cluster_id].append(phrase)
    
    # Filter noise and name categories
    final_categories = {}
    for cid, phrases in category_map.items():
        if cid == -1 or len(phrases) < 3: 
            continue
            
        # Find most representative phrase
        cat_embeds = self.embedder.encode(phrases)
        centroid = np.mean(cat_embeds, axis=0)
        sims = [1 - cosine(e, centroid) for e in cat_embeds]
        cat_name = phrases[np.argmax(sims)]
        
        final_categories[cat_name] = phrases
    
    return final_categories
