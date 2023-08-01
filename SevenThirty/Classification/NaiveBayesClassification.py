import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample email data
emails = [
    "Buy our amazing products at a discount!",
    "Limited time offer: free shipping on all orders",
    "Check out this cool gadget",
    "Congratulations! You won a prize",
    "Get rich quick with our investment plan",
    "Help a prince in Nigeria transfer funds",
    "Hi, how are you doing?",
    "Please find the attached report"
 ]

# Corresponding labels (1 for spam, 0 for not spam)
labels = np.array([1, 1, 1, 1, 1, 1, 0, 0])

# Create a CountVectorizer to convert text data to numerical feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Create a Naive Bayes classifier and train it
classifier = MultinomialNB()
classifier.fit(X, labels)

while (0 < 1):

    userinput = input("Enter input: ")    
    
    new_email_vector = vectorizer.transform(list(userinput))

    prediction = classifier.predict(new_email_vector)
    
    if prediction[0] == 1:
        print("This email is spam.")
    else:
        print("This email is not spam.")
