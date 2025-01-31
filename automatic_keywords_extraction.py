def _discover_categories_and_phrases(self, texts, num_phrases):
    """Automatically discover both key phrases and their semantic categories"""
    # Step 1: Extract candidate phrases using YAKE
    kw_extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.8, top=num_phrases*2)
    candidates = []
    for text in texts:
        keywords = kw_extractor.extract_keywords(text)
        candidates.extend([kw[0].lower() for kw in keywords])
    
    # Step 2: Cluster phrases into categories using semantic embeddings
    phrase_embeddings = self.embedder.encode(list(set(candidates)))
    
    # Use HDBSCAN for density-based clustering
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
    clusters = clusterer.fit_predict(phrase_embeddings)
    
    # Step 3: Name clusters using most central phrases
    from sklearn.metrics import pairwise_distances
    category_map = {}
    for cluster_id in np.unique(clusters):
        if cluster_id == -1: continue  # Ignore noise
        
        mask = clusters == cluster_id
        cluster_phrases = np.array(candidates)[mask]
        cluster_embeddings = phrase_embeddings[mask]
        
        # Find most central phrase
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = pairwise_distances(cluster_embeddings, [centroid])
        central_phrase = cluster_phrases[np.argmin(distances)]
        
        category_map[cluster_id] = {
            'name': central_phrase,
            'phrases': cluster_phrases.tolist()
        }
    
    return category_map





