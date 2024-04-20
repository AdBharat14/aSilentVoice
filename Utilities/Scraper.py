import requests
from bs4 import BeautifulSoup

def download_manga(url, output_folder):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract image URLs (modify this based on the website's structure)
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags]

    # Create a folder for the manga
    os.makedirs(output_folder, exist_ok=True)

    # Download images
    for i, img_url in enumerate(img_urls):
        img_response = requests.get(img_url)
        img_path = os.path.join(output_folder, f"page_{i + 1}.png")
        with open(img_path, 'wb') as img_file:
            img_file.write(img_response.content)



base_url = 'https://a-silent-voice.online/manga/a-silent-voice-chapter-'
num_chapters = 30  # Set the total number of chapters

for chapter_number in range(11, num_chapters + 1):
    chapter_url = f'{base_url}{chapter_number}'
    # response = requests.get(chapter_url)
    # soup = BeautifulSoup(response.content, 'html.parser')

    # Extract relevant data (modify this based on your requirements)
    # For example, extract titles, images, or other content

    # Your scraping code here...
    # ...
    output_directory = 'The_silent_voice'
    download_manga(chapter_url, output_directory+'/'+str(chapter_number))

    print(f"Scraped data from Chapter {chapter_number}: {chapter_url}")

# You can save or process the scraped data as needed
