python img-align --ref "checkboxes/big_images/21543780.png" --img "checkboxes/big_images/*.png" --out "checkboxes/aligned"

Rscript checkboxes/file_df.R
python checkboxes/crop_checkboxes.py
Rscript checkboxes/process_checkboxes.R
python checkboxes/process_cropped_checkboxes.py

set TF_CPP_MIN_LOG_LEVEL=1
python checkboxes/fit_checkbox_model.py
python checkboxes/predict_on_uncoded.py
Rscript checkboxes/pivot_predictions.R
