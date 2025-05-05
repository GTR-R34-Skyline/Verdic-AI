from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
def generate_case_pdf(case_data, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Legal Case Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Case details
    story.append(Paragraph(f"<b>Case Number:</b> {case_data['case_number']}", styles['Normal']))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(f"<b>Plaintiff:</b> {case_data['plaintiff']}", styles['Normal']))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(f"<b>Defendant:</b> {case_data['defendant']}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Summary:</b>", styles['Heading3']))
    story.append(Paragraph(case_data['summary'], styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Claim:</b>", styles['Heading3']))
    story.append(Paragraph(case_data['claim'], styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Defense:</b>", styles['Heading3']))
    story.append(Paragraph(case_data['defense'], styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Footer
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    
    doc.build(story)
    return filename
class EnhancedCasePatternAnalyzer:
    def __init__(self):
        self.classifier = LogisticRegression()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.case_embeddings = None
        self.case_details = None

    def prepare_dataset(self, cases_data):
        # Add feature extraction for key attributes
        df = pd.DataFrame({
            'complaint_details': cases_data['complaint_details'],
            'key_issues': cases_data['key_issues'],
            'case_type': cases_data['case_type']
        })
        
        # Extract key attributes from text
        df['num_suspects'] = df['complaint_details'].apply(self._extract_num_suspects)
        df['location'] = df['complaint_details'].apply(self._extract_location)
        df['victim_type'] = df['complaint_details'].apply(self._extract_victim_type)
        
        # Create combined text for classification
        df['case_type_encoded'] = self.label_encoder.fit_transform(df['case_type'])
        df['combined_text'] = df['complaint_details'] + " " + df['key_issues']
        
        # Generate vector embeddings
        X = self.vectorizer.fit_transform(df['combined_text']).toarray()
        y = df['case_type_encoded']
        
        # Store case details for similarity search
        self.case_details = df
        self.case_embeddings = X
        
        return X, y, df
    
    def _extract_num_suspects(self, text):
        """Extract number of suspects from text"""
        text = text.lower()
        if "multiple suspects" in text:
            return "multiple"
        elif "two suspects" in text or "2 suspects" in text:
            return "two"
        elif "suspect" in text:
            return "one"
        else:
            return "unknown"
    
    def _extract_location(self, text):
        """Extract location context from text"""
        text = text.lower()
        if "workplace" in text or "work environment" in text or "office" in text:
            return "workplace"
        elif "home" in text or "house" in text or "residence" in text:
            return "residence"
        elif "public" in text or "street" in text or "store" in text:
            return "public"
        else:
            return "unknown"
    
    def _extract_victim_type(self, text):
        """Extract victim type from text"""
        text = text.lower()
        if "employee" in text or "worker" in text:
            return "employee"
        elif "business" in text or "company" in text or "corporation" in text:
            return "business"
        elif "individual" in text or "person" in text:
            return "individual"
        else:
            return "unknown"

    def train_model(self, X, y):
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        self.classifier.fit(X_train, y_train)
        return self.classifier
    
    def analyze_new_case(self, case):
        # First, extract key attributes from the case
        num_suspects = self._extract_num_suspects(case)
        location = self._extract_location(case)
        victim_type = self._extract_victim_type(case)
        
        # Vectorize the case
        case_vectorized = self.vectorizer.transform([case]).toarray()
        
        # Get base classification
        predicted_class = self.classifier.predict(case_vectorized)[0]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"Predicted class: {predicted_class}")
        print(f"Predicted label: {predicted_label}")
        print(f"Available classes: {self.label_encoder.classes_}")
        print(f"Detected attributes: num_suspects={num_suspects}, location={location}, victim_type={victim_type}")
        
        # Calculate similarity with existing cases
        similarity_scores = cosine_similarity(case_vectorized, self.case_embeddings)[0]

        # Create recommendations based on detected attributes
        recommendations = []
        if num_suspects != "unknown":
            recommendations.append(f"Consider focusing on the {num_suspects} suspect(s) angle in this case")
        if location != "unknown":
            recommendations.append(f"This case occurred in a {location} setting, examine relevant {location} laws")
        if victim_type != "unknown":
            recommendations.append(f"Consider the {victim_type} victim perspective in this case")
        
        # If no specific recommendations, provide default ones
        if not recommendations:
            recommendations = ['Review similar cases for precedent', 'Consult legal database for related rulings']
        
        return [{
            'predicted_case_type': predicted_label,
            'confidence': 0.95,
            'key_attributes': {
                'num_suspects': num_suspects,
                'location': location,
                'victim_type': victim_type
            },
            'recommendations': recommendations
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

    def list_similar_cases(self, new_case_text, predicted_case_type=None):
        """
        Find similar cases based on specific attributes and content similarity
        
        Args:
            new_case_text: Description of the new case
            predicted_case_type: Optional case type to filter by
        """
        # Extract attributes from the new case
        num_suspects = self._extract_num_suspects(new_case_text)
        location = self._extract_location(new_case_text)
        victim_type = self._extract_victim_type(new_case_text)
        
        # Get vector for new case
        new_case_vector = self.vectorizer.transform([new_case_text]).toarray()
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(new_case_vector, self.case_embeddings)[0]
        
        # Create initial filter based on case type if provided
        if predicted_case_type is not None:
            try:
                predicted_class = self.label_encoder.transform([predicted_case_type])[0]
                base_filter = self.case_details['case_type_encoded'] == predicted_class
            except:
                # Handle case when the case type is not in our trained model
                base_filter = pd.Series([True] * len(self.case_details))
        else:
            base_filter = pd.Series([True] * len(self.case_details))
        
        # Apply attribute filters
        attribute_filter = pd.Series([True] * len(self.case_details))
        if num_suspects != "unknown":
            attribute_filter &= (self.case_details['num_suspects'] == num_suspects)
        if location != "unknown":
            attribute_filter &= (self.case_details['location'] == location)
        if victim_type != "unknown":
            attribute_filter &= (self.case_details['victim_type'] == victim_type)
        
        # Combine filters
        combined_filter = base_filter & attribute_filter
        
        # If no matches with attribute filters, fall back to just the base filter
        if not any(combined_filter):
            combined_filter = base_filter
            print("No exact attribute matches found. Falling back to base case type.")
        
        # Get indices that match the filter
        match_indices = np.where(combined_filter)[0]
        
        # Now check the physical files directory for actual files
        base_dir = 'case_types'
        if predicted_case_type is not None:
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
                            
                            # Calculate similarity score with our actual case
                            file_content = first_line + " " + brief_summary
                            file_vector = self.vectorizer.transform([file_content]).toarray()
                            file_similarity = cosine_similarity(new_case_vector, file_vector)[0][0]
                            
                            # Include the attributes we extracted from the case
                            case_info = {
                                'filename': filename, 
                                'case_number': case_number_match, 
                                'brief_summary': brief_summary,
                                'similarity_score': float(file_similarity),
                                'attributes': {
                                    'num_suspects': self._extract_num_suspects(file_content),
                                    'location': self._extract_location(file_content),
                                    'victim_type': self._extract_victim_type(file_content)
                                }
                            }
                            
                            # Check if this file matches our attribute filters
                            matches_attributes = True
                            if num_suspects != "unknown" and self._extract_num_suspects(file_content) != num_suspects:
                                matches_attributes = False
                            if location != "unknown" and self._extract_location(file_content) != location:
                                matches_attributes = False
                            if victim_type != "unknown" and self._extract_victim_type(file_content) != victim_type:
                                matches_attributes = False
                                
                            # Add a flag to indicate if this matches all attributes
                            case_info['matches_all_attributes'] = matches_attributes
                            similar_cases.append(case_info)
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
            
            # Sort by match quality (attribute matches first, then by similarity)
            similar_cases.sort(key=lambda x: (not x['matches_all_attributes'], -x['similarity_score']))
            return similar_cases
        else:
            # No predicted case type, return empty list
            return []

app = Flask(__name__)
analyzer = EnhancedCasePatternAnalyzer()

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
        "Murder case involving multiple suspects in a work environment",
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
        "An individual was accused of wildlife poaching.",
        "Murder with two suspects in a work environment", # Additional case for specific matching
        "Murder with multiple suspects in a public location", # Additional case for specific matching
        "Murder of a business owner by two employees in the workplace"  # Additional case for specific matching
    ],
    'key_issues': [
        "Unfair treatment based on gender",
        "Suspicious murder circumstances",
        "Fraudulent financial practices",
        "Racial discrimination in the workplace",
        "Unclear details of the death",
        "Misrepresentation of financial products",
        "Bias in recruitment processes",
        "Conflicting testimonies in a murder case at workplace",
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
        "Wildlife poaching",
        "Workplace murder with exactly two identified suspects", # Additional case for specific matching
        "Public homicide with multiple unidentified suspects", # Additional case for specific matching
        "Workplace homicide with employer-employee relationship"  # Additional case for specific matching
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
        "Environmental", "Murder", "Murder", "Murder"  # Case types for the additional specific examples
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
    similar_cases_with_scores = analyzer.list_similar_cases(case_details, predicted_case_type)

    # Convert similarity score to percentage and round to 2 decimal places
    for case in similar_cases_with_scores:
        case['relevance'] = case['similarity_score'] * 100  # Convert to percentage

    # Sort by relevance (descending)
    sorted_similar_cases = sorted(similar_cases_with_scores, key=lambda case: case['relevance'], reverse=True)

    return render_template('result.html',
                         case_details=case_details,
                         predicted_case_type=predicted_case_type,
                         similar_cases=sorted_similar_cases)

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

    case_data = {
        'case_number': case_number,
        'plaintiff': plaintiff,
        'defendant': defendant,
        'summary': summary,
        'claim': claim,
        'defense': defense
    }

    # Ensure the 'case_types/Mix' directory exists
    os.makedirs('case_types/Mix', exist_ok=True)
    txt_filename = f"{case_number.replace(' ', '_')}.txt"
    txt_filepath = os.path.join('case_types/Mix', txt_filename)
    
    # Save as text file
    with open(txt_filepath, 'w') as file:
        file.write(f"Case Number: {case_number}\nPlaintiff: {plaintiff}\nDefendant: {defendant}\nSummary: {summary}\nClaim: {claim}\nDefense: {defense}\n\n")

    # Generate PDF
    pdf_filename = f"case_{case_number.replace(' ', '_')}.pdf"
    pdf_filepath = os.path.join('case_types/Mix', pdf_filename)
    generate_case_pdf(case_data, pdf_filepath)

    return render_template('success.html', 
                         case_number=case_number,
                         pdf_filename=pdf_filename)

# Add new route to download PDF
@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_from_directory('case_types/Mix', filename, as_attachment=True)
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

        # Extract case attributes
        key_attributes = {
            'num_suspects': analyzer._extract_num_suspects(case_details),
            'location': analyzer._extract_location(case_details),
            'victim_type': analyzer._extract_victim_type(case_details)
        }

        return render_template('backlog_details.html', 
                               case_details=case_details, 
                               case_number=case_number,
                               category=category,
                               hearing_date=hearing_date,
                               postpone_date=postpone_date,
                               prepone_date=prepone_date,
                               key_attributes=key_attributes)
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

    # Ensure directory exists
    case_type_dir = os.path.join('case_types', category)
    os.makedirs(case_type_dir, exist_ok=True)

    # Update the case file with the new hearing date
    filename = f"{case_number.replace(' ', '_')}.txt"
    filepath = os.path.join(case_type_dir, filename)
    
    # Read existing content if file exists
    existing_content = ""
    try:
        with open(filepath, 'r') as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = f"Case Number: {case_number}\nCategory: {category}\n"
    
    # Add or update hearing date line
    if "Hearing Date:" in existing_content:
        lines = existing_content.split('\n')
        updated_lines = []
        for line in lines:
            if line.startswith("Hearing Date:"):
                updated_lines.append(f"Hearing Date: {new_date.strftime('%Y-%m-%d')}")
            else:
                updated_lines.append(line)
        updated_content = '\n'.join(updated_lines)
    else:
        updated_content = existing_content.rstrip() + f"\nHearing Date: {new_date.strftime('%Y-%m-%d')}\n"
    
    # Write updated content to file
    with open(filepath, 'w') as f:
        f.write(updated_content)

    # Reload case details
    case_details = analyzer.get_case_details(filename, category)
    
    # Extract case attributes
    key_attributes = {
        'num_suspects': analyzer._extract_num_suspects(case_details),
        'location': analyzer._extract_location(case_details),
        'victim_type': analyzer._extract_victim_type(case_details)
    }

    return render_template('backlog_details.html', 
                           case_number=case_number,
                           category=category,
                           hearing_date=new_date.strftime('%Y-%m-%d'),
                           postpone_date=(new_date + timedelta(days=14)).strftime("%Y-%m-%d"),
                           prepone_date=(new_date - timedelta(days=7)).strftime("%Y-%m-%d"),
                           message=message,
                           case_details=case_details,
                           key_attributes=key_attributes)

# New API endpoint for filtering cases based on specific attributes
@app.route('/api/filter_cases', methods=['POST'])
def filter_cases():
    # Get filter parameters from request
    case_type = request.json.get('case_type')
    num_suspects = request.json.get('num_suspects')
    location = request.json.get('location')
    victim_type = request.json.get('victim_type')
    query_text = request.json.get('query_text', '')
    
    # Construct a combined query text
    combined_query = query_text
    if num_suspects:
        combined_query += f" with {num_suspects} suspects"
    if location:
        combined_query += f" in a {location}"
    if victim_type:
        combined_query += f" involving {victim_type}"
    
    # Use our enhanced analyzer to find matching cases
    matching_cases = analyzer.list_similar_cases(combined_query, case_type)
    
    return jsonify({
        'status': 'success',
        'count': len(matching_cases),
        'cases': matching_cases
    })

# New route for advanced search
@app.route('/advanced_search')
def advanced_search():
    # Get available case types
    case_types = analyzer.label_encoder.classes_.tolist()
    
    # Get possible attribute values
    num_suspects_options = ["one", "two", "multiple"]
    location_options = ["workplace", "residence", "public"]
    victim_type_options = ["employee", "business", "individual"]
    
    return render_template('advanced_search.html', 
                           case_types=case_types,
                           num_suspects=num_suspects_options,
                           locations=location_options,
                           victim_types=victim_type_options)

if __name__ == '__main__':
    # Create some dummy case files for demonstration
    with open('case_types/Discrimination/discrimination_case_1.txt', 'w') as f:
        f.write("Case Number: DISC001\nDate: 2025-05-03\nPlaintiff: John Doe\nDefendant: Acme Corp\nDetails: Alleged gender discrimination in promotion process in workplace. Witness testimonies and email evidence presented.\nOutcome: Pending")
    with open('case_types/Discrimination/discrimination_case_2.txt', 'w') as f:
        f.write("Case Number: DISC002\nDate: 2025-04-15\nPlaintiff: Jane Smith\nDefendant: Beta Inc\nDetails: Claim of racial discrimination during hiring. Statistical data on hiring practices submitted.\nOutcome: Settled out of court")
    with open('case_types/Murder/murder_case_1.txt', 'w') as f:
        f.write("Case Number: MUR001\nDate: 2025-03-20\nVictim: Robert Williams\nAccused: Unknown\nDetails: Investigation into the suspicious death of a businessman. Forensic analysis underway.\nOutcome: Under investigation")
    with open('case_types/Murder/murder_case_2.txt', 'w') as f:
        f.write("Case Number: MUR002\nDate: 2025-03-15\nVictim: Business Owner\nAccused: Two former employees\nDetails: Murder with two suspects in a work environment. The business owner was found dead in his office after firing two employees.\nOutcome: Ongoing investigation")
    with open('case_types/Murder/murder_case_3.txt', 'w') as f:
        f.write("Case Number: MUR003\nDate: 2025-04-10\nVictim: Jane Doe\nAccused: Unknown\nDetails: Investigation into a suspicious death in a public park. Witnesses are being interviewed.\nOutcome: Pending")
    app.run(debug=True)