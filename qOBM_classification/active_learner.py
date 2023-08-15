import numpy as np
import copy
from qOBM_classification.mask_generation import get_mask_generator, get_all_viable_masks
from qOBM_classification.feature_extraction import (add_all_manual_features, add_all_masks_boxes, 
                                                    get_manual_feature_array, get_class_array, add_class_label,
                                                    get_all_keys_idxs, add_all_phasor_boxes)
from qOBM_classification.visualize import label_masks_class
from qOBM_classification.utils import random_sample_list, parition_list
from qOBM_classification.object_dataset import make_phasor_dataset, make_phasor_dataset_inference

def filter_out_ignore_index(X, y, ignore_index=3):
    # Create a boolean mask where y is not equal to 3
    mask = (y != ignore_index)

    # Use the mask to filter both X and y arrays
    filtered_X = X[mask]
    filtered_y = y[mask]

    return filtered_X, filtered_y

class ActiveLearner:
    def __init__(self, image_file_paths, phasor_file_paths, sam_checkpoint, query_seed=None,
                 random_sample=20, min_samples=500, max_samples=1000, early_termination_samples=10):
        
        self.query_seed = query_seed
        self.random_sample = random_sample
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.early_termination_samples = early_termination_samples
        
        mask_generator = get_mask_generator(sam_checkpoint)
        
        all_masks = get_all_viable_masks(mask_generator, image_file_paths, phasor_file_paths,
                                 min_area=3000, max_area=15000, min_roundness=0.85, progress=True)

        add_all_manual_features(all_masks)
        add_all_masks_boxes(all_masks, bbox_extend=150)

        keys_idxs = get_all_keys_idxs(all_masks)
        self.all_masks = all_masks
        self.keys_idxs = keys_idxs
        self.total_num_masks = len(keys_idxs)
        
        self.train_keys_idxs = []
        
    def set_model(self, model=None):
        """
        sklearn interface model with `fit`, `predict`, and `predict_proba`. 
        If None, uses random forest with standard scaler.
        """
        if model is None:
            model = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
        self.model = model
        
    def learn(self):
        # first iteration query
        query_parition = random_sample_list(self.keys_idxs, 
                                            sampled_num=self.random_sample, 
                                            seed=self.query_seed)
        self.queried_keys_idxs, self.val_keys_idxs = query_parition
        
        terminate = False
        while not terminate:
            # label
            label_masks_class(self.all_masks, self.queried_keys_idxs)
            self.train_keys_idxs += self.queried_keys_idxs
            
            # next query
            X = get_manual_feature_array(self.all_masks, self.train_keys_idxs)
            X_val = get_manual_feature_array(self.all_masks, self.val_keys_idxs)
            y = get_class_array(self.all_masks, self.train_keys_idxs)
            X, y = filter_out_ignore_index(X, y)
            self.model.fit(X, y)
            y_proba = self.model.predict_proba(X_val)
            confidences = y_proba.max(axis=1)
            
            # gradually increase confidence threshold to avoid empty query
            queried_idxs = np.array([])
            confidence_threshold = 0.5
            while queried_idxs.size == 0:
                queried_idxs = np.arange(len(confidences))[confidences < confidence_threshold]
                confidence_threshold += 0.05
                
            # confidence and random threshold
            self.queried_keys_idxs, val_keys_idxs = parition_list(self.val_keys_idxs, queried_idxs)
            random_query_keys_idxs, self.val_keys_idxs = random_sample_list(val_keys_idxs, 
                                                                  sampled_num=self.random_sample, 
                                                                  seed=self.query_seed)
            self.queried_keys_idxs += random_query_keys_idxs
            
            # termination criteria
            if len(self.train_keys_idxs) > self.min_samples:
                less_than_min_samples_queried = len(self.queried_keys_idxs) < self.early_termination_samples + self.random_sample
                more_than_max_samples_labeled = len(self.train_keys_idxs) > self.max_samples
                terminate = less_than_min_samples_queried or more_than_max_samples_labeled

        # after termination
        self.val_keys_idxs += self.queried_keys_idxs
        print("num labeled masks:", len(self.train_keys_idxs))
        print("num unlabeled masks:", len(self.val_keys_idxs))
        
    def predict(self, all_masks, val_keys_idxs):
        X_val = get_manual_feature_array(all_masks, val_keys_idxs)
        y_pred = self.model.predict(X_val)
        out_all_masks = copy.deepcopy(all_masks)
        add_class_label(out_all_masks, val_keys_idxs, y_pred)
        return out_all_masks
        
    def predict_unlabeled(self):
        self.predict(self.all_masks, self.val_keys_idxs)
        
