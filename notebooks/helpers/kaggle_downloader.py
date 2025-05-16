import kagglehub
from pathlib import Path
import shutil

class KaggleDatasetManager:

    def __init__(self, dataset_name: str, target_folder: str = './data'):
        """
        Inicjalizacja menadżera pobierania.
        :param dataset_name: Nazwa zestawu danych z KaggleHub (np. "arezaei81/heartcsv")
        :param target_folder: Folder docelowy (domyślnie './data')
        """
        self.dataset_name = dataset_name
        self.target_folder = Path(target_folder)

    
    def download_and_prepare(self) -> Path:
        """
        Pobiera dane z KaggleHub, kopiuje do folderu projektu i usuwa pliki tymczasowe.
        :return: Ścieżka do folderu z przygotowanymi danymi
        """
        print(f"Downloading dataset: {self.dataset_name}")

        original_path = kagglehub.dataset_download(self.dataset_name)

        self.target_folder.mkdir(parents=True, exist_ok=True)

        for file in Path(original_path).iterdir():
            shutil.copy(file, self.target_folder)

        print("Files copied to:", self.target_folder)

        for file in Path(original_path).iterdir():
            file.unlink()
        Path(original_path).rmdir()

        print(f'Original files from kagglehub cache removed')

        return self.target_folder

