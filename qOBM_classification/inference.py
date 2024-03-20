import os
import pickle
from io import BytesIO
from PIL import Image
from qOBM_classification.mask_generation import cv2_read_image, get_mask_generator, get_all_viable_masks
from qOBM_classification.feature_extraction import (add_all_manual_features, get_manual_feature_array, 
                                                    add_class_label, get_all_keys_idxs)
from qOBM_classification.utils import phase2GRAY
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Inference:
    """
    Currently uses SAM and random forest.
    """
    def __init__(self, predictor_cfg, predictor_weights):
        self.predictor_cfg = predictor_cfg # currently unused
        self.predictor_weights = predictor_weights
        
        assert("sam_checkpoint" in predictor_weights)
        assert(os.path.isdir(checkpoint_path := predictor_weights["sam_checkpoint"]))
        assert("random_forest_data_pkl" in predictor_weights)
        assert(os.path.isdir(rf_pkl_path := predictor_weights["random_forest_data_pkl"]))
        
        with open(rf_pkl_path, 'rb') as handle:
            rf_data = pickle.load(handle)
        self.rf_model = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
        self.rf_model.fit(rf_data["X"], rf_data["y"])
        self.mask_generator = get_mask_generator(checkpoint_path)
        
    def predict(self, phasor):
        """
        Expect image of shape (h,w,3) as float np.ndarray
        """
        phase = phasor[:,:,0]
        gray = phase2GRAY(phase, vmin=-0.2, vmax=0.5)
        img = Image.fromarray(gray , 'L')
        image_file_path = BytesIO()
        img.save(image_file_path, format='PNG')
        processed_image = cv2_read_image(image_file_path)
        
        # predict
        all_masks = get_all_viable_masks(self.mask_generator, [processed_image], [phasor],
                                         min_area=3000, max_area=15000, min_roundness=0.85, 
                                         progress=False, in_memory=True)
        add_all_manual_features(all_masks)
        keys_idxs = get_all_keys_idxs(all_masks)
        X_test = get_manual_feature_array(all_masks, keys_idxs)
        y_pred = self.rf_model.predict(X_test)
        add_class_label(all_masks, keys_idxs, y_pred)

        (_, v), = all_masks.items()
        live_cells = 0
        dead_cells = 0
        for mask in v["masks"]:
            c = mask["class"]
            if c == 1:
                live_cells += 1
            elif c == 2:
                dead_cells += 1
        viability = live_cells / (live_cells + dead_cells)
        phenotype_distribution = [0,] # placeholder
        
        return {"img_vis": processed_image,
                "viability": viability,
                "phenotype_distribution": phenotype_distribution}
    