import pandas as pd                                # For reading and handling Excel data
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text descriptions into vector form
from sklearn.metrics.pairwise import cosine_similarity      # To calculate similarity between vectors
import re                                         # Regular expressions for input format validation
import os                                         # To check file existence

# --- Minimal ADK Framework Mock ---

class Agent:
    def __init__(self):
        pass  # Base Agent class (empty here, just a placeholder for ADK structure)

def intent_handler(intent_name):
    # Decorator function for intent handling (mock implementation)
    def decorator(func):
        return func
    return decorator

def run_agent(agent_class):
    # Function to instantiate the agent and run demo tests
    
    agent = agent_class()  # Create an instance of the provided agent class
    print("Agent started.\n")

    # Demo test input: list of HSN codes to validate
    test_codes = ["01", "01011010", "9999", "abc123", "  010110  "]
    print("=== Validation Test ===")
    result = agent.validate_hsn_handler({"hsn_codes": test_codes})  # Call validation intent
    print(result)  # Print results

    # Demo test input: description query for HSN code suggestion
    print("\n=== Suggestion Test ===")
    suggestions = agent.suggest_hsn_handler({"query": "horses"})  # Call suggestion intent
    print(suggestions)  # Print suggestions

# --- HSN Data and Agent Implementation ---

class HSNData:
    def __init__(self, file_path):
        # Check if the Excel file exists before loading
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"HSN data file not found: {file_path}")

        self.df = pd.read_excel(file_path)  # Load Excel data into a pandas DataFrame

        # Convert HSNCode column to string and remove any leading/trailing spaces
        self.df["HSNCode"] = self.df["HSNCode"].astype(str).str.strip()

        # Fill any missing descriptions with empty string and strip spaces
        self.df["Description"] = self.df["Description"].fillna("").str.strip()

        # Store all HSN codes in a set for O(1) existence lookup
        self.code_set = set(self.df["HSNCode"])

        # Create a TF-IDF vectorizer and transform all descriptions into vectors
        self.vectorizer = TfidfVectorizer()
        self.desc_vectors = self.vectorizer.fit_transform(self.df["Description"])

    def validate_code(self, code):
        code = code.strip()  # Remove whitespace from input code

        # Check format: code must be 2 to 8 digits exactly
        if not re.fullmatch(r"\d{2,8}", code):
            return {"valid": False, "reason": "Invalid format. Must be 2-8 digits."}

        # Check existence in the dataset
        if code not in self.code_set:
            return {"valid": False, "reason": "Code not found in dataset."}

        # Retrieve matching record from DataFrame
        result = self.df[self.df["HSNCode"] == code].iloc[0]

        # Return valid status and description
        return {"valid": True, "description": result["Description"]}

    def hierarchical_check(self, code):
        """
        Check if parent HSN codes exist.
        For code like '01011010':
        parent codes are every 2 digits prefix: '01', '0101', '010110'
        """
        code = code.strip()
        
        # Generate all even-length prefixes except full code
        levels = [code[:i] for i in range(2, len(code), 2)]

        # Find which parent codes exist in the dataset
        found = [lvl for lvl in levels if lvl in self.code_set]
        missing = [lvl for lvl in levels if lvl not in self.code_set]

        return found, missing

    def suggest_codes(self, query, top_k=3):
        query = query.strip()
        if not query:
            return []  # Return empty list if query is empty

        # Transform query text into vector using the existing TF-IDF vectorizer
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity of query vector to all description vectors
        similarities = cosine_similarity(query_vec, self.desc_vectors).flatten()

        # Sort indices of descriptions by similarity (descending order)
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Optional: filter out results below a similarity threshold (e.g., 0.1)
        min_sim_threshold = 0.1
        filtered_indices = [i for i in top_indices if similarities[i] >= min_sim_threshold]

        if not filtered_indices:
            return []  # No sufficiently similar results

        # Get top matching HSN codes and descriptions as list of dictionaries
        suggestions = self.df.iloc[filtered_indices][["HSNCode", "Description"]]
        return suggestions.to_dict("records")


class HSNValidationAgent(Agent):
    def __init__(self, file_path=None):
        super().__init__()
        # Use default file path if none provided
        if file_path is None:
            file_path = r"C:\Data Science and AI\Project\HSN Code Validation and Suggestion Agent\HSN_SAC.xlsx"
        self.data = HSNData(file_path)  # Load HSN data into the agent

    @intent_handler("validate_hsn")
    def validate_hsn_handler(self, inputs):
        codes = inputs.get("hsn_codes", [])
        if not codes:
            return {"error": "No HSN codes provided."}

        response = []
        for code in codes:
            result = self.data.validate_code(code)
            if result["valid"]:
                # If valid, also check hierarchical parent codes
                found, missing = self.data.hierarchical_check(code)
                response.append({
                    "code": code.strip(),
                    "valid": True,
                    "description": result["description"],
                    "hierarchy_found": found,
                    "hierarchy_missing": missing
                })
            else:
                # If invalid, return reason
                response.append({
                    "code": code.strip(),
                    "valid": False,
                    "reason": result["reason"]
                })
        return {"results": response}

    @intent_handler("suggest_hsn")
    def suggest_hsn_handler(self, inputs):
        query = inputs.get("query", "").strip()
        if not query:
            return {"error": "No query provided."}

        suggestions = self.data.suggest_codes(query)
        if not suggestions:
            return {"message": "No suggestions found for your query."}

        return {"suggestions": suggestions}


# --- Run the agent ---

if __name__ == "__main__":
    run_agent(HSNValidationAgent)
