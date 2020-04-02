
def get_indices(dataset,class_name):
    indices =  []
    try : 
        labels = dataset.labels
    except :
        labels = dataset.targets
    for i in range(len(labels)):
        if labels[i] == class_name:
            indices.append(i)
    return indices