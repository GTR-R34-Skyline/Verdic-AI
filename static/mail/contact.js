// File: /c:/Users/shash/Verdic-AI/static/mail/contact.js

// Function to send a contact email
async function sendContactEmail(name, email, message) {
    try {
        const response = await fetch('/api/contact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, message }),
        });

        if (!response.ok) {
            throw new Error('Failed to send contact email');
        }

        const result = await response.json();
        console.log('Email sent successfully:', result);
        return result;
    } catch (error) {
        console.error('Error sending contact email:', error);
        throw error;
    }
}

// Example usage
// sendContactEmail('John Doe', 'john.doe@example.com', 'Hello, this is a test message.');