def cleanup_folders(folder_path1, folder_path2):
    """
    This function checks all the files in folder 1 and removes
      elements from folder2 that have different names."""

    import os 
    import shutil
    # Get the list of files in folder 1
    csv_files = [f for f in os.listdir(folder_path1) if f.endswith('.csv')]

    # Get the list of files in folder 2
    folders = os.listdir(folder_path2)
    # Loop through each file in folder 1
    for folder_name in folders:
        if folder_name + ".csv" not in csv_files:
            # If the file is not in folder 1, remove it from folder 2
            folder_to_remove = os.path.join(folder_path2, folder_name)
            if os.path.isdir(folder_to_remove):
                shutil.rmtree(folder_to_remove)
                print(f"Removed folder: {folder_to_remove}")
            else:
                print(f"Not a directory: {folder_to_remove}")
   
        else:
            print(f"Keeping folder: {folder_name}")

cleanup_folders('./test/results', './test/session')