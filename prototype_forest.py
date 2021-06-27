class Node:
    def __init__(self):

        self.right = None
        self.left = None
        
        self.prototype = None
        
        self.column = None
        self.threshold = None
        
        self.probas = None
        self.depth = None
        
        self.is_terminal = False

class PrototypeTreeClassifier:
    def __init__(self, 
                 train_features,
                 feature_types = ["min", "mean", "max"], 
                 max_depth = 3, 
                 min_samples_leaf = 1, 
                 min_samples_split = 2, 
                 prototype_count = 1,
                 use_prototype_learner = True):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.prototype_count = prototype_count
        self.feature_types = feature_types
        self.use_prototype_learner = use_prototype_learner
        self.Tree = None
        self.train_features = train_features
        
    def prototype(self, bags, features, labels, prototype_count):
        if self.use_prototype_learner:
            prototypes = find_prototype(bags, features, labels)
            check = prototypes.cpu().detach().numpy()

            check.resize(check.shape[1], check.shape[2])
            
            return check
        
        else:
            number_of_rows = self.train_features.shape[0]
            random_indices = np.random.choice(number_of_rows, 
                                              size=prototype_count, 
                                              replace=False)
        
            return self.train_features[random_indices, :]
            
    def nodeProbas(self, y):
        # for each unique label calculate the probability for it
        probas = []

        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def features_via_prototype(self, feature_types, features, bag_ids, prototypes):
        distances = self.calculate_distances(features, prototypes)

        bin_count  = np.unique(bag_ids, return_counts=True)[1]
        ids, index  = np.unique(bag_ids, return_index=True)
        feature_list = []

        for i in range(0, prototypes.shape[0]):
            if "max" in feature_types:
                group_max = np.maximum.reduceat(distances[:, i], index)
                max_vals = np.repeat(group_max, bin_count)
                feature_list.append(max_vals)

            if "min" in feature_types:
                group_min = np.minimum.reduceat(distances[:, i], index)
                min_vals = np.repeat(group_min, bin_count)
                feature_list.append(min_vals)

            if "mean" in feature_types:
                group_sum = np.add.reduceat(distances[:, i], index)
                group_mean = group_sum/bin_count
                mean_vals = np.repeat(group_mean, bin_count)
                feature_list.append(mean_vals)

        return np.array(np.transpose(feature_list))
    
    def dist1d(self, features, prototypes, distance_type="l2"):
        if distance_type == "l2":
            distance = np.linalg.norm(features - prototypes, axis=1)
        elif distance_type == "l1":
            distance = np.abs(features - prototypes)
            distance = np.sum(distance, axis=1)
        
        return distance

    def calculate_distances(self, features, prototypes):
        feature_list = []
        for i in range(0, prototypes.shape[0]):
            data = self.dist1d(features, prototypes[i], distance_type="l2")
            feature_list.append(data)
        data = np.column_stack(feature_list)

        return data

    def calcBestSplit(self, features, features_updated, labels, bag_ids):
        bdc = tree.DecisionTreeClassifier(random_state=0, 
                                  max_depth=1, 
                                  criterion="entropy",
                                  min_samples_split=2)
        bdc.fit(features_updated, labels.flatten())
        
        threshold = bdc.tree_.threshold[0]
        split_col = bdc.tree_.feature[0]

        features_left = features[features_updated[:,split_col] <= bdc.tree_.threshold[0]]
        features_right = features[features_updated[:,split_col] > bdc.tree_.threshold[0]]
        
        labels_left = labels[features_updated[:,split_col] <= bdc.tree_.threshold[0]]
        labels_right = labels[features_updated[:,split_col] > bdc.tree_.threshold[0]]

        bag_ids_left = bag_ids[features_updated[:,split_col] <= bdc.tree_.threshold[0]]
        bag_ids_right = bag_ids[features_updated[:,split_col] > bdc.tree_.threshold[0]]

        return split_col, threshold, features_left, features_right, labels_left, labels_right, bag_ids_left, bag_ids_right
    
    def buildDT(self, features, labels, bag_ids, node):
            '''
            Recursively builds decision tree from the top to bottom
            '''
            # checking for the terminal conditions

            if node.depth >= self.max_depth:
                node.is_terminal = True
                return

            if features.shape[0] < self.min_samples_split:
                node.is_terminal = True
                return

            if np.unique(labels).shape[0] == 1:
                node.is_terminal = True
                return
            
            node.prototype = self.prototype(bag_ids, features, labels, self.prototype_count)
            features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, node.prototype)
            # calculating current split
            (splitCol, 
             thresh, 
             features_left, 
             features_right, 
             labels_left, 
             labels_right, 
             bag_ids_left, 
             bag_ids_right) = self.calcBestSplit(features, 
                                                 features_updated, 
                                                 labels, 
                                                 bag_ids)

            if splitCol is None:
                node.is_terminal = True
                return
            
            if features_left.shape[0] < self.min_samples_leaf or features_right.shape[0] < self.min_samples_leaf:
                node.is_terminal = True
                return

            node.column = splitCol
            node.threshold = thresh

            # creating left and right child nodes
            node.left = Node()
            node.left.depth = node.depth + 1
            node.left.probas = self.nodeProbas(labels_left)

            node.right = Node()
            node.right.depth = node.depth + 1
            node.right.probas = self.nodeProbas(labels_right)

            # splitting recursevely

            self.buildDT(features_right, labels_right, bag_ids_right, node.right)
            self.buildDT(features_left, labels_left, bag_ids_left, node.left)
            
    def fit(self, features, labels, bag_ids):
        '''
        Standard fit function to run all the model training
        '''
        self.classes = np.unique(labels)
        
        self.Tree = Node()
        self.Tree.depth = 1
        
        self.buildDT(features, labels, bag_ids, self.Tree)
    
    def predictSample(self, features, bag_ids, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''
       
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas
        
        features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, node.prototype)

        if features_updated[0][node.column] > node.threshold:
            probas = self.predictSample(features, bag_ids, node.right)
        else:
            probas = self.predictSample(features, bag_ids, node.left)
            
        return probas
    
    def predict(self, features, bag_ids):
        '''
        Returns the labels for each X
        '''
        if type(features) == pd.DataFrame:
            X = np.asarray(features)
                
        sort_index = np.argsort(bag_ids)
        bag_ids = bag_ids[sort_index]
        features = features[sort_index]
    
        features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, self.Tree.prototype)
        
        index  = np.unique(bag_ids, return_index=True)[1]
        count  = np.unique(bag_ids, return_counts=True)[1]
        index = np.append(index, bag_ids.shape[0])   
        predictions = []
        
        for i in range(0, len(index) - 1):
            pred = np.argmax(self.predictSample(features[index[i]:index[i+1]], 
                                                bag_ids[index[i]:index[i+1]], 
                                                self.Tree))
            pred = np.repeat(pred, count[i])
            predictions = np.concatenate((predictions, pred), axis=0)
        
        return np.asarray(predictions)       
        
class PrototypeForest:
    def __init__(self, size,
                feature_types = ["min", "mean", "max"], 
                max_depth = 3, min_samples_leaf = 2, min_samples_split = 2, stratified = True, sample_rate = 0.8,
                prototype_count = 1,
                use_prototype_learner = True):
        self.size = size
        self._trees = []
        self._tuning_trees = []
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.stratified = stratified
        self.sample_rate = sample_rate
        self.prototype_count = prototype_count
        self.use_prototype_learner = use_prototype_learner
        
    def sample(self, features, labels, bag_ids, stratified, sample_rate):
        if stratified:
            pos_sample_size = math.ceil(np.where(labels == 1)[0].shape[0] * sample_rate)
            neg_sample_size = math.ceil(np.where(labels == 0)[0].shape[0] * sample_rate)
            indices_pos = np.random.choice(np.where(labels == 1)[0], pos_sample_size, replace=False)
            indices_neg = np.random.choice(np.where(labels == 0)[0], neg_sample_size, replace=False)
            inbag_indices = np.concatenate((indices_pos, indices_neg))
        else:
            sample_size = math.ceil(labels.shape[0] * sample_rate)
            inbag_indices = np.random.choice(np.where(labels == 1)[0], sample_size, replace=False)
        
        oo_bag_mask = np.ones(labels.shape[0], dtype=bool)
        oo_bag_mask[inbag_indices] = False

        outbag_indices = np.where(oo_bag_mask == 1)

        return inbag_indices, outbag_indices
    
    def fit(self, features, labels, bag_ids):
        for i in range(self.size):
            if self.use_prototype_learner:
                print(f"Tree {i} will be trained")
            tree = PrototypeTreeClassifier(max_depth=self.max_depth, 
                                           min_samples_leaf=self.min_samples_leaf, 
                                           min_samples_split=self.min_samples_split,
                                          prototype_count = self.prototype_count,
                                          use_prototype_learner = self.use_prototype_learner,
                                          train_features = features)
            
            (inbag_indices,
             outbag_indices) = self.sample(features, labels, bag_ids, self.stratified, self.sample_rate)      
            
            tree.inbag_indices = inbag_indices
            tree.outbag_indices = outbag_indices
            
            inbag_features = features[inbag_indices]
            inbag_labels = labels[inbag_indices]
            inbag_bag_ids = bag_ids[inbag_indices]
                                    
            tree.fit(inbag_features, inbag_labels, inbag_bag_ids)

            self._trees.append(tree)

    def predict(self, features, bag_ids):
        temp = [t.predict(features, bag_ids) for t in self._trees]
        preds = np.transpose(np.array(temp))
        
        return mode(preds,1)[0]
    
    def predict_proba(self, features, bag_ids):
        temp = [t.predict(features, bag_ids) for t in self._trees]
        preds = np.transpose(np.array(temp))
        
        return np.sum(preds==1, axis=1)/self.size