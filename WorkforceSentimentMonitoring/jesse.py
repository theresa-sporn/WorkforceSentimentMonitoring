def fit_transform_tfidf(self, reduce=10): 
        '''fit and transforms train set, transforms test set. 
        Also reduce the amount of features so text does not have too many features 
        ''' 
    if self.features_train is None: 
        t0 = time() 
        print('Running train_test_split first with default values') 
        self.train_test_split() 
        print(f"Complete... fitting and transforming: {round(time()-t0, 3)}s") 
    else: 
        t0 = time() 
        self.features_train = self.vectorizer.fit_transform(self.features_train) #partial_fit 
        self.features_test = self.vectorizer.transform(self.features_test) 
        self.vocabulary = self.vectorizer.get_feature_names() 
        print(f"Fit and transform complete time: {round(time()-t0, 3)}s") 
         
    #from sklearn.feature_selection import SelectPercentile, f_classif 
    selector = SelectPercentile(f_classif, percentile=reduce) # ch2 = SelectKBest(chi2) 
    self.selector_fitted = 