import os
import csv

def update_csv_with_folder_files(folder_path, csv_path):
    # Lire les fichiers déjà listés dans le CSV (si le fichier existe)
    existing_paths = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_paths.add(row['Path'])
    
    # Parcourir tous les fichiers dans le dossier et ses sous-dossiers
    new_entries = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            if full_path not in existing_paths:
                file_type = os.path.splitext(file)[1][1:]  # extension sans le '.'
                file_size = round(os.path.getsize(full_path) / (1024**3), 3)  # taille en GB arrondie à 3 décimales
                new_entries.append({'Path': full_path, 'Type': file_type, 'Size': file_size})
    
    if not new_entries:
        print("Aucun nouveau fichier à ajouter.")
        return
    
    # Écriture dans le CSV (création ou ajout)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Path', 'Type', 'Size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        for entry in new_entries:
            writer.writerow(entry)

# Exemple d'utilisation
folder = r"C:\SynologyDrive"
csv_file = r"output.csv"
update_csv_with_folder_files(folder, csv_file)
