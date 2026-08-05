import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Configuration
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

EXPRESSIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==================== FUNCTIONS ====================
def check_dataset_structure(data_dir):
    """
    Check and display the dataset structure.
    Returns True if both 'train' and 'test' folders exist and contain class subfolders; otherwise False.
    """
    print("="*60)
    print("Checking Dataset Structure")
    print("="*60)

    ok = True
    for split in ['train', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"Missing '{split}' folder at: {split_path}")
            ok = False
            continue

        print(f"\n{split.upper()} folder: {split_path}")
        total = 0
        classes = []
        for entry in sorted(os.listdir(split_path)):
            emotion_path = os.path.join(split_path, entry)
            if os.path.isdir(emotion_path):
                count = len([f for f in os.listdir(emotion_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {entry}: {count} images")
                total += count
                classes.append(entry)
        print(f"  Total: {total} images")
        if total == 0:
            print(f"  Warning: No images found in '{split}' folder.")
            ok = False

    if ok:
        print("\n✓ Dataset structure looks OK.")
    else:
        print("\n✗ Dataset structure problems detected. Please fix paths / folders.")
    return ok

def create_data_generators(data_dir, batch_size=BATCH_SIZE):
    """
    Create tf.data datasets for training/validation/test using GPU-friendly tf.image ops.
    Returns: train_ds, val_ds, test_ds, class_names
    """
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test folder not found: {test_path}")

    AUTOTUNE = tf.data.AUTOTUNE
    # Use image_dataset_from_directory which yields (images, labels)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        shuffle=False,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        shuffle=False
    )

    class_names = train_ds.class_names

    # Preprocessing / augmentation using tf.image (runs on GPU when possible)
    def preprocess_train(image_batch, label_batch):
        # images come as uint8 [0,255], convert to float32 and scale
        image_batch = tf.cast(image_batch, tf.float32) / 255.0
        # random augmentations
        image_batch = tf.image.random_flip_left_right(image_batch)
        image_batch = tf.image.random_brightness(image_batch, 0.1)
        image_batch = tf.image.random_contrast(image_batch, 0.9, 1.1)
        # optionally random rotation (CPU fallback) - use small rotation via tf.image.rot90 when needed
        return image_batch, label_batch

    def preprocess_val(image_batch, label_batch):
        image_batch = tf.cast(image_batch, tf.float32) / 255.0
        return image_batch, label_batch

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE)

    # Cache + prefetch for performance
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    # Try to copy dataset to GPU device to shift preprocessing/execution there when a GPU is available.
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        try:
            gpu_device = gpus[0].name  # e.g. '/device:GPU:0'
            train_ds = train_ds.apply(tf.data.experimental.copy_to_device(gpu_device))
            train_ds = train_ds.apply(tf.data.experimental.prefetch_to_device(gpu_device))
            val_ds = val_ds.apply(tf.data.experimental.copy_to_device(gpu_device))
            val_ds = val_ds.apply(tf.data.experimental.prefetch_to_device(gpu_device))
            test_ds = test_ds.apply(tf.data.experimental.copy_to_device(gpu_device))
            test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device(gpu_device))
            print(f"✓ Datasets copied to device {gpu_device}")
        except Exception as e:
            print(f"Could not copy datasets to GPU: {e}")

    # Print counts (compute via directory scan like before)
    def count_images(folder):
        total = 0
        for cls in os.listdir(folder):
            cls_path = os.path.join(folder, cls)
            if os.path.isdir(cls_path):
                total += len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        return total

    train_count = count_images(train_path)
    # validation is taken from train_path split 20%
    val_count = int(train_count * 0.2)
    test_count = count_images(test_path)

    print(f"\n✓ Data datasets created:")
    print(f"  Training samples: {train_count}")
    print(f"  Validation samples: {val_count}")
    print(f"  Test samples: {test_count}")
    print(f"  Classes found: {class_names}")

    return train_ds, val_ds, test_ds, class_names

def build_model():
    """
    Build CNN model for expression classification
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Increased learning rate from 1e-4 to 1e-3
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    """
    Train the model with callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_fer2013_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history

def evaluate_model(model, test_ds, class_names):
    """
    Evaluate model on test dataset and show results
    """
    print("\n" + "="*60)
    print("Evaluating Model on Test Set")
    print("="*60)

    results = model.evaluate(test_ds, verbose=1)
    # model.evaluate on dataset returns list [loss, acc]
    if isinstance(results, list) and len(results) >= 2:
        test_loss, test_acc = results[0], results[1]
    else:
        # fallback
        test_loss, test_acc = results, None

    if test_acc is not None:
        print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✓ Test Loss: {test_loss:.4f}")

    # Get predictions and true labels
    print("\nGenerating predictions...")
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)

    # collect true labels from dataset
    y_true_list = []
    for _, labels in test_ds.unbatch().batch(1024):
        y_true_list.append(np.argmax(labels.numpy(), axis=1))
    if y_true_list:
        y_true = np.concatenate(y_true_list, axis=0)
    else:
        y_true = np.array([], dtype=np.int32)

    # Classification report
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    plt.close()

    return test_acc, cm

def plot_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history.get('accuracy', []), label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history.get('val_accuracy', []), label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history.get('loss', []), label='Train Loss', linewidth=2)
    axes[1].plot(history.history.get('val_loss', []), label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ Training history saved as 'training_history.png'")
    plt.show()
    plt.close()

def predict_single_image(model, image_path, class_names=None):
    """
    Predict expression from a single image
    """
    if class_names is None:
        class_names = EXPRESSIONS

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

    predictions = model.predict(img_normalized, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])

    # Display image with prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Show original image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Show prediction probabilities
    probs = predictions[0]
    # Ensure labels match probabilities length
    labels = class_names if len(class_names) == len(probs) else [f"Class {i}" for i in range(len(probs))]
    ax2.barh(labels, probs)
    ax2.set_xlabel('Probability')
    ax2.set_title('Expression Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.show()
    plt.close()

    return labels[predicted_class], confidence

def visualize_sample_predictions(model, test_ds, class_names, num_samples=16):
    """
    Visualize sample predictions from test dataset
    """
    # take one batch
    for images, labels in test_ds.take(1):
        imgs = images.numpy()
        labs = labels.numpy()
        break

    num_available = min(num_samples, imgs.shape[0])
    preds = model.predict(imgs[:num_available])
    cols = min(4, num_available)
    rows = (num_available + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = np.asarray(axes).reshape(-1)

    for i in range(num_available):
        ax = axes[i]
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        true_label = class_names[np.argmax(labs[i])]
        pred_label = class_names[np.argmax(preds[i])]
        confidence = np.max(preds[i])
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})', color=color, fontsize=9)
        ax.axis('off')

    for j in range(num_available, axes.size):
        try:
            axes[j].axis('off')
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300)
    print("✓ Sample predictions saved as 'sample_predictions.png'")
    plt.show()
    plt.close()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("FER-2013 FACIAL EXPRESSION RECOGNITION")
    print("="*60)

    # Set your dataset path here
    DATA_DIR = os.path.dirname(__file__)  # points to folder that contains train/ and test/

    # Step 1: Check dataset structure
    ok = check_dataset_structure(DATA_DIR)
    if not ok:
        print("Exiting due to dataset structure problems.")
        sys.exit(1)

    # Step 2: Create data generators
    print("\n" + "="*60)
    print("Creating Data Generators")
    print("="*60)
    try:
        train_ds, val_ds, test_ds, class_names = create_data_generators(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error creating data generators: {e}")
        sys.exit(1)

    # Step 3: Build model
    print("\n" + "="*60)
    print("Building CNN Model")
    print("="*60)
    model = build_model()
    print(f"\nTotal parameters: {model.count_params():,}")

    # Step 4: Train model
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS)

    # Step 5: Plot training history
    print("\n" + "="*60)
    print("Plotting Training History")
    print("="*60)
    plot_history(history)

    # Step 6: Evaluate on test set
    test_acc, cm = evaluate_model(model, test_ds, class_names)

    # Step 7: Visualize sample predictions
    print("\n" + "="*60)
    print("Visualizing Sample Predictions")
    print("="*60)
    visualize_sample_predictions(model, test_ds, class_names)

    # Step 8: Save final model
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    model.save('fer2013_final_model.keras')
    print("✓ Model saved as 'fer2013_final_model.keras'")

    print("\n" + "="*60)
    print("TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print("\nFiles created:")
    print("  📁 best_fer2013_model.h5 (best model)")
    print("  📁 fer2013_final_model.h5 (final model)")
    print("  📊 training_history.png")
    print("  📊 confusion_matrix.png")
    print("  📊 sample_predictions.png")

    # Example usage
    print("\n" + "="*60)
    print("HOW TO USE THE TRAINED MODEL:")
    print("="*60)
    print("""
# Load the model
from tensorflow import keras
model = keras.models.load_model('best_fer2013_model.keras')

# Predict on a single image
expression, confidence = predict_single_image(model, 'your_image.jpg')
print(f"Predicted: {expression} ({confidence*100:.1f}%)")
    """)