import os
from pathlib import Path
import tensorflow as tf

BASE = Path('d:/seaaai')
MODEL_SRC = BASE / 'model_repository' / 'covid_classifier' / '1' / 'attention_resnet_covid_classifier.keras'
VERSION_DIR = BASE / 'model_repository' / 'covid_classifier' / '1'

print('Source model:', MODEL_SRC)
print('Saving SavedModel into:', VERSION_DIR)

if not MODEL_SRC.exists():
    print('ERROR: source model not found')
    raise SystemExit(1)

# Try loading model
try:
    model = tf.keras.models.load_model(str(MODEL_SRC), compile=False, safe_mode=False)
    print('Loaded model via tf.keras.models.load_model')
except Exception as e:
    print('tf.keras.models.load_model failed:', e)
    # Try rebuilding architecture similar to original
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

    base = ResNet50(input_shape=(224,224,3), include_top=False, weights=None)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    try:
        model.load_weights(str(MODEL_SRC))
        print('Rebuilt architecture and loaded weights')
    except Exception as e2:
        print('Failed to rebuild weights load:', e2)
        raise

# Save as SavedModel
saved_model_dir = VERSION_DIR / 'saved_model'
saved_model_path = str(saved_model_dir)
print('Saving to:', saved_model_path)
saved_model_dir.mkdir(parents=True, exist_ok=True)
try:
    tf.saved_model.save(model, saved_model_path)
    print('SavedModel saved successfully')
except Exception as e:
    print('Failed to save SavedModel:', e)
    raise

# Remove python backend file if present
python_backend = VERSION_DIR / 'model.py'
if python_backend.exists():
    try:
        python_backend.unlink()
        print('Removed python backend file:', python_backend)
    except Exception as e:
        print('Failed to remove python backend file:', e)

print('Conversion complete')
