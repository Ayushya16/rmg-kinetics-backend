import numpy as np

def build_feature_vector_from_row(input_data, features_list: list, scaler=None):
    """
    Converts input (either a dict or list) into a numpy array ordered according to features_list.
    If scaler is provided, applies scaling.
    """

    # --- Handle both dict and list inputs ---
    if isinstance(input_data, dict):
        x = []
        for f in features_list:
            v = input_data.get(f, 0.0)
            try:
                x.append(float(v))
            except:
                x.append(0.0)
    elif isinstance(input_data, list):
        # If the user directly passes a list of numbers
        x = [float(v) for v in input_data]
    else:
        raise ValueError("Input must be a dictionary or list of feature values.")

    # --- Convert to NumPy array ---
    arr = np.array(x).reshape(1, -1)

    # --- Apply scaler if available ---
    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception as e:
            print(f"⚠️ Skipping scaler due to mismatch: {e}")

    return arr
