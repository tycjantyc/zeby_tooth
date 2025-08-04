import os

def validation_spliter(lista, validation_mode = True, validation_split = 0.05):
    
    split_index = int(len(lista) * validation_split)
    
    if validation_mode:
        return lista[:split_index]
    else:
        return lista[split_index:]

def init_agh(data_directory):

    lista = []

    path_cbct = os.path.join(data_directory, 'db1_v1.0_nifti')
    path_labels = os.path.join(data_directory, 'db1_v1.0_nifti_teeth_masks_single_cls_weak')
    path_labels_final = os.path.join(data_directory, 'db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak')
       
    paths_cbct = sorted(os.listdir(path_cbct), key=lambda x: int(x.split('.')[0]))   
    files_labels = sorted(os.listdir(path_labels), key=lambda x: int(x.split('.')[0]))
    files_labels_final = sorted(os.listdir(path_labels_final), key=lambda x: int(x.split('.')[0]))

    paths_cbct = paths_cbct[1:len(files_labels)+1]
        
    for file_cbct, file_label, file_label_final in zip(paths_cbct, files_labels, files_labels_final):
            
        file_cbct = os.path.join(path_cbct, file_cbct, 'image.nii.gz')
        file_label = os.path.join(path_labels, file_label)
        file_label_final = os.path.join(path_labels_final, file_label_final)    
        lista.append((file_cbct, file_label, file_label_final))

    return lista

def init_canal_snippets(data_directory):
    
    lista = []

    path_cbct = os.path.join(data_directory, 'snippets_img')
    path_labels = os.path.join(data_directory, 'snippets_mask')

    files_cbct = sorted(os.listdir(path_cbct))
    files_labels = sorted(os.listdir(path_labels))
         
    for file_cbct, file_label in zip(files_cbct, files_labels):
            
        file_cbct = os.path.join(path_cbct, file_cbct)
        file_label = os.path.join(path_labels, file_label)
        lista.append((file_cbct, file_label))

    return lista

def init_tooth_fairy(data_directory):
    
    lista = []

    path_cbct = os.path.join(data_directory, 'imagesTr')
    path_labels = os.path.join(data_directory, 'labelsTr')
       
    files_cbct = sorted(os.listdir(path_cbct))
    files_labels = sorted(os.listdir(path_labels))
         
    for file_cbct, file_label_1, file_label_2 in zip(files_cbct, files_labels, files_labels):
            
        file_cbct = os.path.join(path_cbct, file_cbct)
        file_label_1 = os.path.join(path_labels, file_label_1)
        file_label_2 = os.path.join(path_labels, file_label_2)
        lista.append((file_cbct, file_label_1, file_label_2))

    return lista