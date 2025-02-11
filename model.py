import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class CasePatternAnalyzer:
    def __init__(self):
        self.classifier = LogisticRegression()  # Replace with your model
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def prepare_dataset(self, cases_data):
        # Convert the dataset into a DataFrame
        df = pd.DataFrame({
            'complaint_details': cases_data['complaint_details'],
            'key_issues': cases_data['key_issues'],
            'case_type': cases_data['case_type']
        })

        # Encode the case types
        df['case_type_encoded'] = self.label_encoder.fit_transform(df['case_type'])

        # Combine text features into a single feature for vectorization
        df['combined_text'] = df['complaint_details'] + " " + df['key_issues']

        # Vectorize the combined text
        X = self.vectorizer.fit_transform(df['combined_text']).toarray()  # Convert to array
        y = df['case_type_encoded']
        return X, y, df

    def train_model(self, X, y):
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        # Get only the present labels in y_test
        present_classes = np.unique(y_test)
        target_names = self.label_encoder.inverse_transform(present_classes)  # Map to original labels


        
        return self.classifier



    def analyze_new_case(self, case):
        case_vectorized = self.vectorizer.transform([case]).toarray()
        predicted_class = self.classifier.predict(case_vectorized)[0]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"Predicted class: {predicted_class}")
        print(f"Predicted label: {predicted_label}")
        print(f"Available classes: {self.label_encoder.classes_}")
        
        return [{
            'predicted_case_type': predicted_label,
            'confidence': 0.95,  # Dummy confidence
            'recommendations': ['Recommendation 1', 'Recommendation 2']
        }]
    
    def list_similar_cases(self,predicted_case_type):
        # Define the base directory where case types are stored
        base_dir = 'case_types'
        
        # Construct the path to the directory for the predicted case type
        case_type_dir = os.path.join(base_dir, predicted_case_type)
        
        # Check if the directory exists
        if not os.path.exists(case_type_dir):
            print(f"No cases found for the case type: {predicted_case_type}")
            return []

        # List all text files in the directory
        similar_cases = []
        for filename in os.listdir(case_type_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(case_type_dir, filename)
                # Read the case details from the text file
                with open(file_path, 'r') as file:
                    case_details = file.read()
                    similar_cases.append((filename, case_details))

        return similar_cases

def main():
    cases_data = {
        'complaint_details': [
            "Gender-based discrimination in workplace promotion",
            "Murder of business tycoon by business partner",
            "Stock market fraud through manipulation",
            "Employee faced discrimination due to race",
            "Suspicious circumstances surrounding a death",
            "Fraudulent activities in a financial institution",
            "Gender discrimination in hiring practices",
            "Murder case involving multiple suspects",
            "Fraudulent investment schemes targeting seniors",
            "A teenager was caught shoplifting a candy bar from a convenience store.",
            "A city council member is accused of accepting bribes from a construction company.",
            "An employee was wrongfully terminated after reporting safety violations.",
            "A woman was charged with petty theft for stealing a bicycle.",
            "A minor was caught vandalizing public property.",
            "A person was arrested for public intoxication and disorderly conduct",
            "Long-standing workplace discrimination involving multiple employees",
            "Complex financial fraud spanning multiple organizations",
            "Sophisticated murder investigation with intricate evidence",
            "Systematic corruption in a government department",
            "Potential discrimination with subtle workplace dynamics",
            # New Cases
            "Suspicious financial transactions suggest potential investment fraud.",
            "A man was arrested for driving under the influence.",
            "A woman was arrested for shoplifting.",
            "A man was arrested for driving without a license.",
            "High-profile politician accused of accepting illegal campaign funds.",
            "Illegal land acquisition through fraudulent documents.",
            "Property dispute between two neighboring families.",
            "Unauthorized use of copyrighted material for commercial gain.",
            "An online marketplace accused of selling counterfeit products.",
            "A hacker breached a financial institution's security system.",
            "A driver caused an accident while texting.",
            "A doctor performed a surgery without proper consent.",
            "A patient sued a hospital for medical malpractice.",
            "A social media influencer falsely advertised a product.",
            "Leaking of confidential government data to foreign entities.",
            "Unauthorized cloning of a famous software product.",
            "A couple was in a legal battle over property inheritance.",
            "A taxi driver hit a pedestrian due to reckless driving.",
            "Illegal drug trafficking across international borders.",
            "A person was wrongly accused of identity theft.",
            "Bribery scandal involving law enforcement officers.",
            "A large-scale Ponzi scheme affecting thousands of investors.",
            "Alleged political conspiracy leading to an unlawful arrest.",
            "Unauthorized deepfake content causing reputational damage.",
            "A journalist was accused of defamation against a corporation.",
            "Illegal dumping of toxic waste by a factory.",
            "An individual was accused of wildlife poaching."
        ],
        'key_issues': [
            "Unfair treatment based on gender",
            "Suspicious murder circumstances",
            "Fraudulent financial practices",
            "Racial discrimination in the workplace",
            "Unclear details of the death",
            "Misrepresentation of financial products",
            "Bias in recruitment processes",
            "Conflicting testimonies in a murder case",
            "Scams affecting elderly individuals",
            "Theft of minor goods",
            "Bribery and abuse of power",
            "Retaliation against whistleblowers",
            "Theft of personal property",
            "Destruction of public property",
            "Disorderly conduct in public",
            "Systemic bias in promotion and compensation",
            "Multi-layered financial manipulation",
            "Forensic evidence and witness testimony complexities",
            "Institutional corruption and power abuse",
            "Nuanced workplace interpersonal conflicts",
            # New Key Issues
            "Suspicious financial transactions",
            "Driving under the influence",
            "Shoplifting incident",
            "Driving without a license",
            "Illegal campaign funding",
            "Fraudulent land acquisition",
            "Property dispute between families",
            "Unauthorized copyright use",
            "Selling counterfeit products",
            "Hacking a financial institution",
            "Accident caused by texting",
            "Surgery without consent",
            "Medical malpractice lawsuit",
            "False advertisement of a product",
            "Leaking confidential data",
            "Cloning software products",
            "Property inheritance dispute",
            "Reckless driving incident",
            "Drug trafficking",
            "Wrongful identity theft accusation",
            "Bribery involving law enforcement",
            "Ponzi scheme",
            "Political conspiracy",
            "Deepfake content issue",
            "Defamation against a corporation",
            "Illegal toxic waste dumping",
            "Wildlife poaching"
        ],
        'case_type': [
            'Discrimination', 'Murder', 'Fraud', 'Discrimination', 'Murder', 
            'Fraud', 'Discrimination', 'Murder', 'Fraud', 'Petty Case', 
            'Corruption', 'Discrimination', 'Petty Case', 'Petty Case', 
            'Petty Case', 'Discrimination', 'Fraud', 'Murder', 'Corruption', 
            'Discrimination',
            # New Case Types
            "Scam", "Petty Case", "Petty Case", "Petty Case", "Corruption",
            "Land Disputes", "Land Disputes", "Copyrights", "Cybercrime",
            "Cybercrime", "Car", "Medical", "Medical", "Scam", "Cybercrime",
            "Copyrights", "Land Disputes", "Car", "Criminal", "Cybercrime",
            "Corruption", "Scam", "Mix", "Cybercrime", "Mix", "Environmental",
            "Environmental"
        ]
    }
    
    analyzer = CasePatternAnalyzer()
    X, y, dataset = analyzer.prepare_dataset(cases_data)

    # Check the distribution of classes
    class_counts = pd.Series(y).value_counts()

    # Allow all classes
    filtered_indices = dataset.index
    X_filtered = X[filtered_indices]  # Make sure to use correct indexing
    y_filtered = y[filtered_indices]  # Ensure y is also filtered correctly

    # Check if there are any samples left after filtering
    if len(X_filtered) == 0 or len(y_filtered) == 0:
        print("No valid samples left after filtering. Please check your dataset.")
        return



    # Proceed with training the model
    model = analyzer.train_model(X_filtered, y_filtered)

    # Test new cases
    test_cases = [
        "Murder case involving multiple suspects"
    ]
    print("\n--- Case Analysis Results ---")
    for case in test_cases:
        print(f"\nAnalyzing Case: {case}")
        results = analyzer.analyze_new_case(case)
        for result in results:
            print(f"Predicted Case Type: {result['predicted_case_type']}")


            print("similar cases :\n",result['predicted_case_type'])

            predicted_case_type = result['predicted_case_type']

            similar_cases = analyzer.list_similar_cases(predicted_case_type)

            # Print the similar cases
            if similar_cases:
                i=1
                print(f"Similar cases for the predicted case type '{predicted_case_type}':")
                for filename, case_details in similar_cases:
                    print()
                    print(f"Case {i}:")
                    print(f"\nFilename: {filename}")
                    print()
                    print(f"Case Details:{case_details}")
                    i+=1
            else:
                print("No similar cases found.")




if __name__ == "__main__":
    main()