from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class CasePatternAnalyzer:
    def __init__(self):
        self.classifier = LogisticRegression()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def prepare_dataset(self, cases_data):
        df = pd.DataFrame({
            'complaint_details': cases_data['complaint_details'],
            'key_issues': cases_data['key_issues'],
            'case_type': cases_data['case_type']
        })
        df['case_type_encoded'] = self.label_encoder.fit_transform(df['case_type'])
        df['combined_text'] = df['complaint_details'] + " " + df['key_issues']
        X = self.vectorizer.fit_transform(df['combined_text']).toarray()
        y = df['case_type_encoded']
        return X, y, df

    def train_model(self, X, y):
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        self.classifier.fit(X_train, y_train)
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
            'confidence': 0.95,
            'recommendations': ['Recommendation 1', 'Recommendation 2']
        }]

    def get_case_details(self, filename, predicted_case_type):
        base_dir = 'case_types'
        case_type_dir = os.path.join(base_dir, predicted_case_type)
        file_path = os.path.join(case_type_dir, filename)
        try:
            with open(file_path, 'r') as f:
                details = f.read()
            return details
        except FileNotFoundError:
            return None

    def list_similar_cases(self, predicted_case_type):
        base_dir = 'case_types'
        case_type_dir = os.path.join(base_dir, predicted_case_type)

        if not os.path.exists(case_type_dir):
            print(f"No cases found for the case type: {predicted_case_type}")
            return []

        similar_cases = []
        for filename in os.listdir(case_type_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(case_type_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                        first_line = file.readline().strip()
                        brief_summary = ""
                        lines = file.readlines()
                        if lines:
                            brief_summary = " ".join(lines[:2]).strip() # Get first 2 lines as summary
                        case_number_match = filename.split('.')[0] # Extract case number from filename
                        similar_cases.append({'filename': filename, 'case_number': case_number_match, 'brief_summary': brief_summary})
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        return similar_cases

app = Flask(__name__)
analyzer = CasePatternAnalyzer()

# Prepare and train the model when the app starts
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
        "Scam", "Petty Case", "Petty Case", "Petty Case", "Corruption",
        "Land Disputes", "Land Disputes", "Copyrights", "Cybercrime",
        "Cybercrime", "Car", "Medical", "Medical", "Scam", "Cybercrime",
        "Copyrights", "Land Disputes", "Car", "Criminal", "Cybercrime",
        "Corruption", "Scam", "Mix", "Cybercrime", "Mix", "Environmental",
        "Environmental"
    ]
}
X, y, dataset = analyzer.prepare_dataset(cases_data)
model = analyzer.train_model(X, y)

# Dictionary to store case hearing dates
case_dates = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/precedence')
def precedence():
    return render_template('precedence.html')

@app.route('/submit', methods=['POST'])
def submit():
    case_details = request.form.get('case_details')

    results = analyzer.analyze_new_case(case_details)
    predicted_case_type = results[0]['predicted_case_type']
    similar_cases = analyzer.list_similar_cases(predicted_case_type)

    return render_template('result.html',
                           case_details=case_details,
                           predicted_case_type=predicted_case_type,
                           similar_cases=similar_cases)

@app.route('/download_case/<predicted_case_type>/<filename>')
def download_case(predicted_case_type, filename):
    case_details = analyzer.get_case_details(filename, predicted_case_type)
    if case_details:
        doc = SimpleDocTemplate(f"case_details_{filename.split('.')[0]}.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(case_details, styles['Normal']))
        doc.build(story)
        return send_from_directory('.', f"case_details_{filename.split('.')[0]}.pdf", as_attachment=True)
    else:
        return "Case details not found."

@app.route('/document')
def document():
    return render_template('document.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/submit1', methods=['POST'])
def submit1():
    case_number = request.form.get('case_number')
    plaintiff = request.form.get('plaintiff')
    defendant = request.form.get('defendant')
    summary = request.form.get('summary')
    claim = request.form.get('claim')
    defense = request.form.get('defense')

    case_details = f"Case Number: {case_number}\nPlaintiff: {plaintiff}\nDefendant: {defendant}\nSummary: {summary}\nClaim: {claim}\nDefense: {defense}\n\n"

    # Ensure the 'case_types/Mix' directory exists
    os.makedirs('case_types/Mix', exist_ok=True)
    filename = f"{case_number.replace(' ', '_')}.txt"
    filepath = os.path.join('case_types/Mix', filename)
    with open(filepath, 'w') as file:
        file.write(case_details)

    return render_template('success.html', case_number=case_number)

@app.route('/backlog')
def backlog():
    return render_template('backlog.html')

@app.route('/get_case_details', methods=['POST'])
def get_case_details():
    case_number = request.form.get('case_number')
    category = request.form.get('category')

    # Generate a case filename from the case number
    filename = f"{case_number.replace(' ', '_')}.txt"

    # Check if the case exists
    case_details = analyzer.get_case_details(filename, category)

    if case_details:
        # Generate a random date in the next 30 days if not already in our dictionary
        if case_number not in case_dates:
            today = datetime.now()
            random_days = np.random.randint(5, 30)
            hearing_date = today + timedelta(days=random_days)
            case_dates[case_number] = hearing_date.strftime("%Y-%m-%d")
        
        hearing_date = case_dates[case_number]
        # Calculate postpone and prepone dates
        postpone_date = (datetime.strptime(hearing_date, "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
        prepone_date = (datetime.strptime(hearing_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

        return render_template('backlog_details.html', 
                               case_details=case_details, 
                               case_number=case_number,
                               category=category,
                               hearing_date=hearing_date,
                               postpone_date=postpone_date,
                               prepone_date=prepone_date)
    else:
        return render_template('backlog.html', error=f"Case {case_number} not found in category {category}")

@app.route('/update_hearing', methods=['POST'])
def update_hearing():
    case_number = request.form.get('case_number')
    action = request.form.get('action')
    current_date = datetime.strptime(request.form.get('current_date'), "%Y-%m-%d")
    category = request.form.get('category')

    if action == 'postpone':
        new_date = current_date + timedelta(days=14)
        message = f"Hearing has been postponed to {new_date.strftime('%Y-%m-%d')}."
    elif action == 'prepone':
        new_date = current_date - timedelta(days=7)
        if new_date < datetime.now():
            new_date = datetime.now() + timedelta(days=1)
        message = f"Hearing has been preponed to {new_date.strftime('%Y-%m-%d')}."
    elif action == 'stay':
        new_date = current_date
        message = f"Hearing remains scheduled on {new_date.strftime('%Y-%m-%d')}."
    else:
        return "Invalid action", 400

    # Update the case date in the in-memory dictionary
    case_dates[case_number] = new_date.strftime("%Y-%m-%d")

    # Update the case file with the new hearing date
    filename = f"case_types/{category}/{case_number}.txt"
    with open(filename, 'w') as f:
        f.write(f"Case Number: {case_number}\nCategory: {category}\nHearing Date: {new_date.strftime('%Y-%m-%d')}\n")

    # Reload case details (assuming it's a function you already have)
    case_details = analyzer.get_case_details(f"{case_number.replace(' ', '_')}.txt", category)

    return render_template('backlog_details.html', 
                           case_number=case_number,
                           category=category,
                           hearing_date=new_date.strftime('%Y-%m-%d'),
                           postpone_date=(new_date + timedelta(days=14)).strftime("%Y-%m-%d"),
                           prepone_date=(new_date - timedelta(days=7)).strftime("%Y-%m-%d"),
                           message=message,
                           case_details=case_details)

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('case_types/Discrimination', exist_ok=True)
    os.makedirs('case_types/Murder', exist_ok=True)
    os.makedirs('case_types/Fraud', exist_ok=True)
    os.makedirs('case_types/Petty Case', exist_ok=True)
    os.makedirs('case_types/Corruption', exist_ok=True)
    os.makedirs('case_types/Scam', exist_ok=True)
    os.makedirs('case_types/Land Disputes', exist_ok=True)
    os.makedirs('case_types/Copyrights', exist_ok=True)
    os.makedirs('case_types/Cybercrime', exist_ok=True)
    os.makedirs('case_types/Car', exist_ok=True)
    os.makedirs('case_types/Medical', exist_ok=True)
    os.makedirs('case_types/Criminal', exist_ok=True)
    os.makedirs('case_types/Mix', exist_ok=True)
    os.makedirs('case_types/Environmental', exist_ok=True)

    # Create some dummy case files for demonstration
    with open('case_types/Discrimination/discrimination_case_1.txt', 'w') as f:
        f.write("Case Number: DISC001\nDate: 2025-05-03\nPlaintiff: John Doe\nDefendant: Acme Corp\nDetails: Alleged gender discrimination in promotion process. Witness testimonies and email evidence presented.\nOutcome: Pending")
    with open('case_types/Discrimination/discrimination_case_2.txt', 'w') as f:
        f.write("Case Number: DISC002\nDate: 2025-04-15\nPlaintiff: Jane Smith\nDefendant: Beta Inc\nDetails: Claim of racial discrimination during hiring. Statistical data on hiring practices submitted.\nOutcome: Settled out of court")
    with open('case_types/Murder/murder_case_1.txt', 'w') as f:
        f.write("Case Number: MUR001\nDate: 2025-03-20\nVictim: Robert Williams\nAccused: Unknown\nDetails: Investigation into the suspicious death of a businessman. Forensic analysis underway.\nOutcome: Under investigation")
    with open('case_types/Fraud/fraud_case_1.txt', 'w') as f:
        f.write("Case Number: FRD001\nDate: 2025-02-10\nComplainant: Investors Group\nAccused: Global Finance Ltd\nDetails: Allegations of a large-scale investment fraud. Financial records being examined.\nOutcome: Ongoing")

    app.run(debug=True)