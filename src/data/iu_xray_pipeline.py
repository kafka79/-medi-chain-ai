import os
import tarfile
import requests
import pandas as pd
import xmltodict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

class IUXrayPipeline:
    def __init__(self, data_dir='data'):
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.reports_url = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
        self.images_url = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_data(self):
        """Download reports and images if not existing."""
        for url in [self.reports_url, self.images_url]:
            filename = url.split('/')[-1]
            filepath = os.path.join(self.raw_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        f.write(data)
                        pbar.update(len(data))
                
                print(f"Extracting {filename}...")
                with tarfile.open(filepath, 'r:gz') as tar:
                    tar.extractall(path=self.raw_dir)

    def parse_reports(self):
        """Parse XML reports into a DataFrame."""
        report_files = glob.glob(os.path.join(self.raw_dir, 'ecgen-radiology', '*.xml'))
        data = []
        
        print(f"Parsing {len(report_files)} reports...")
        for file in tqdm(report_files):
            with open(file, 'r', encoding='utf-8') as f:
                report_dict = xmltodict.parse(f.read())
                
                report_id = report_dict['IUXray']['@id']
                finds = ""
                impr = ""
                
                # Extract findings and impressions
                sections = report_dict['IUXray']['medline_citation']['article']['abstract']['abstract_text']
                for section in sections:
                    if section['@label'] == 'FINDINGS':
                        finds = section.get('#text', '')
                    elif section['@label'] == 'IMPRESSION':
                        impr = section.get('#text', '')
                
                # Extract image IDs
                images = report_dict['IUXray'].get('parentImage', [])
                if isinstance(images, dict):
                    images = [images]
                
                for img in images:
                    img_id = img['@id']
                    data.append({
                        'report_id': report_id,
                        'findings': finds,
                        'impression': impr,
                        'image_id': img_id,
                        'image_path': os.path.join(self.raw_dir, f"{img_id}.png")
                    })
        
        return pd.DataFrame(data)

    def run(self, seed=42):
        """Full pipeline execution."""
        self.download_data()
        df = self.parse_reports()
        
        # Verify image existence
        df['exists'] = df['image_path'].apply(os.path.exists)
        missing = len(df) - df['exists'].sum()
        if missing > 0:
            print(f"Warning: {missing} images missing from disk.")
        df = df[df['exists']].drop(columns=['exists'])

        # Stratified Split (by report_id to prevent leakage)
        # We need to split unique reports first
        unique_reports = df[['report_id']].drop_duplicates()
        
        train_reports, test_val_reports = train_test_split(
            unique_reports, test_size=0.2, random_state=seed
        )
        val_reports, test_reports = train_test_split(
            test_val_reports, test_size=0.5, random_state=seed
        )
        
        train_df = df[df['report_id'].isin(train_reports['report_id'])]
        val_df = df[df['report_id'].isin(val_reports['report_id'])]
        test_df = df[df['report_id'].isin(test_reports['report_id'])]
        
        print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        train_df.to_csv(os.path.join(self.processed_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, 'test.csv'), index=False)
        
        print("Pipeline complete.")

if __name__ == "__main__":
    pipeline = IUXrayPipeline()
    pipeline.run()
