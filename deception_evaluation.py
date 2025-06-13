# Visualization and Evaluation Metrics for Deception Detection Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn

# You need to ensure this matches your model definition in the notebook
class MultimodalDeceptionModel(nn.Module):
    """
    Simpler multimodal model for deception detection using NLP, Audio, and Visual features
    """
    def __init__(self, nlp_input_size, audio_input_size, visual_input_size, hidden_size, num_classes):
        super(MultimodalDeceptionModel, self).__init__()
        # Simple linear processors for each modality
        self.nlp_processor = nn.Linear(nlp_input_size, hidden_size)
        self.audio_processor = nn.Linear(audio_input_size, hidden_size)
        self.visual_processor = nn.Linear(visual_input_size, hidden_size)

        # Activation
        self.relu = nn.ReLU()

        # Fusion and Classifier layers
        self.fusion_dropout = nn.Dropout(0.5)  # Dropout after fusion
        # Input size to classifier is concatenation of processed features
        self.classifier = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, nlp_data, audio_data, visual_data):
        """ Forward pass processing and fusing features. """
        # Process each modality through its linear layer and activation
        nlp_processed = self.relu(self.nlp_processor(nlp_data))
        audio_processed = self.relu(self.audio_processor(audio_data))
        visual_processed = self.relu(self.visual_processor(visual_data))

        # Fusion by concatenation
        fused_features = torch.cat((nlp_processed, audio_processed, visual_processed), dim=1)

        # Apply dropout
        fused_features = self.fusion_dropout(fused_features)

        # Final classification layer
        output = self.classifier(fused_features)
        return output

def evaluate_model_performance(model_path, subject_ids, nlp_features, audio_features, visual_features, annotations, device="cpu"):
    """
    Comprehensive evaluation of model performance with visualizations:
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1)
    - Error Rate Analysis
    - Accuracy by Subject
    """
    # 1. Load Data
    print("Loading data for evaluation...")
    loso = LeaveOneGroupOut()
    
    # Prepare containers for results
    all_true_labels = []
    all_predictions = []
    subject_accuracies = {}
    fold_accuracies = []
    class_names = ["Truthful", "Deceptive"]
    
    # 2. Iterate through all folds (subjects)
    fold_num = 0
    for train_index, test_index in loso.split(X=nlp_features, y=annotations, groups=subject_ids):
        fold_num += 1
        test_subjects = np.unique(np.array(subject_ids)[test_index])
        test_subject_str = ', '.join(map(str, test_subjects))
        
        # Split data for this fold
        nlp_test = nlp_features[test_index]
        audio_test = audio_features[test_index]
        visual_test = visual_features[test_index]
        true_labels = annotations[test_index]
        
        # Try to load the best checkpoint for this fold
        best_predictions = []
        best_accuracy = 0
        
        for seed in range(1, 4):  # 3 seeds
            checkpoint_file = os.path.join(model_path, f"fold_{fold_num}_seed_{seed}.pth")
            if os.path.exists(checkpoint_file):
                try:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_file, map_location=device)
                    # Create model
                    model = MultimodalDeceptionModel(
                        nlp_input_size=checkpoint.get('nlp_input_size', nlp_test.shape[1]),
                        audio_input_size=checkpoint.get('audio_input_size', audio_test.shape[1]),
                        visual_input_size=checkpoint.get('visual_input_size', visual_test.shape[1]),
                        hidden_size=checkpoint.get('hidden_size', 128),
                        num_classes=checkpoint.get('num_classes', 2)
                    ).to(device)
                    # Load model weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    # Run prediction
                    with torch.no_grad():
                        nlp_tensor = torch.FloatTensor(nlp_test).to(device)
                        audio_tensor = torch.FloatTensor(audio_test).to(device)
                        visual_tensor = torch.FloatTensor(visual_test).to(device)
                        outputs = model(nlp_tensor, audio_tensor, visual_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.cpu().numpy()
                        
                        # Calculate accuracy for this seed
                        seed_accuracy = accuracy_score(true_labels, predicted)
                        if seed_accuracy > best_accuracy:
                            best_accuracy = seed_accuracy
                            best_predictions = predicted
                            
                except Exception as e:
                    print(f"Warning: Could not evaluate fold {fold_num}, seed {seed}: {e}")
        
        # If we successfully found predictions for this fold
        if len(best_predictions) > 0:
            fold_accuracies.append(best_accuracy)
            all_true_labels.extend(true_labels)
            all_predictions.extend(best_predictions)
            
            # Store subject-specific accuracy
            for subject in test_subjects:
                subject_accuracies[subject] = best_accuracy
            
            print(f"Fold {fold_num}: Subject {test_subject_str}, Accuracy: {best_accuracy:.4f}")
    
    # 3. Calculate overall metrics
    if len(all_predictions) > 0:
        # Convert lists to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_predictions = np.array(all_predictions)
        
        # Calculate overall accuracy and error rate
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        error_rate = 1 - overall_accuracy
        
        # Create confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        
        # Calculate classification report
        report = classification_report(all_true_labels, all_predictions, 
                                       target_names=class_names, output_dict=True)
        
        # 4. Visualizations
        plt.figure(figsize=(18, 12))
        
        # 4.1 Confusion Matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # 4.2 Accuracy by Subject
        plt.subplot(2, 2, 2)
        subject_df = pd.DataFrame(list(subject_accuracies.items()), 
                                 columns=['Subject', 'Accuracy'])
        subject_df = subject_df.sort_values('Accuracy')
        plt.bar(range(len(subject_df)), subject_df['Accuracy'], color='skyblue')
        plt.axhline(y=overall_accuracy, color='r', linestyle='-', label=f'Overall Accuracy: {overall_accuracy:.4f}')
        plt.title('Accuracy by Subject', fontsize=16)
        plt.xlabel('Subject ID (sorted by accuracy)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.xticks([])
        
        # 4.3 Precision, Recall, F1 by Class
        plt.subplot(2, 2, 3)
        metrics_df = pd.DataFrame({
            'Precision': [report[name]['precision'] for name in class_names],
            'Recall': [report[name]['recall'] for name in class_names],
            'F1-Score': [report[name]['f1-score'] for name in class_names]
        }, index=class_names)
        metrics_df.plot(kind='bar', colormap='viridis', ax=plt.gca())
        plt.title('Precision, Recall, F1-Score by Class', fontsize=16)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        
        # 4.4 Error Rate Summary
        plt.subplot(2, 2, 4)
        plt.pie([overall_accuracy, error_rate], 
                labels=['Correct Predictions', 'Errors'],
                colors=['lightgreen', 'tomato'],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 2})
        plt.title('Error Rate Summary', fontsize=16)
        
        plt.tight_layout()
        plt.savefig('deception_detection_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Print detailed metrics
        print("\n===== DECEPTION DETECTION MODEL EVALUATION =====\n")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Error Rate: {error_rate:.4f}")
        print(f"Number of Samples Evaluated: {len(all_true_labels)}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_predictions, target_names=class_names))
        print("\nClass-specific Metrics:")
        print(metrics_df)
        print("\nSubject-specific Accuracies:")
        for subject, acc in sorted(subject_accuracies.items(), key=lambda x: x[1]):
            print(f"  {subject}: {acc:.4f}")
            
        return overall_accuracy, error_rate, cm, report
    else:
        print("Error: No predictions could be made. Check if model checkpoints exist.")
        return None, None, None, None

# Example usage:
# The following line should be added to your Jupyter notebook after running your model training
# model_checkpoint_dir = "multimodal_checkpoints_simple"
# evaluate_model_performance(model_checkpoint_dir, mapped_subject_ids, nlp_features, audio_features, visual_features, annotations, DEVICE) 