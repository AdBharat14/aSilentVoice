import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

def read_data(manga_name):
    collection_name = "mangas"
    data_dict = {}
    try:
        # Reference to the collection
        collection_ref = db.collection(collection_name)

        # Query documents where the 'name' field equals the specified value
        query = collection_ref.where('name', '==', manga_name).stream()

        # Print data from each document
        
        for doc in query:
            data_dict = doc.to_dict()
            data_dict['id'] = doc.id
    except Exception as e:
        print("Error:", e)
    
    return data_dict

def create_document(document_data):
    collection_name = "mangas"
    try:
        # Reference to the collection
        collection_ref = db.collection(collection_name)

        # Query documents where the 'name' field equals the specified value
        new_doc_ref = collection_ref.add(document_data)

    except Exception as e:
        print("Error:", e)

def update_data(manga_name , new_data):
    collection_name = "mangas"
    try:
        old_data = read_data(manga_name)
        try:
            # Reference to the collection
            doc_ref = db.collection(collection_name).document(old_data['id'])

            doc_ref.update(new_data)
        except Exception as e:
            print("Error:", e)
    except Exception as e:
        create_document(new_data)
    

def post_data(manga_name , main_text):
    document_data = read_data(manga_name)
    document_data["main_text"] = main_text
    update_data(manga_name , document_data)


def post_summary(manga_name , summary):
    document_data = read_data(manga_name)
    document_data["summary"] = summary
    update_data(manga_name , document_data)

def get_summary(manga_name):
    doc = read_data(manga_name)
    return doc["summary"]