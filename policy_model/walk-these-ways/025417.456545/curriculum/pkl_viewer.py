import pickle

# Load the pickle file
with open('info.pkl', 'rb') as file:
    data = pickle.load(file)

# Assuming 'data' is a dictionary, list, or other serializable structure.
# You might need to adjust the way of handling 'data' based on its structure.
with open('info.txt', 'w') as file:
    file.write(str(data))
