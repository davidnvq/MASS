from mass.mc import MassClassifier

mass_config = [{"name": "MASS", "classifier" : MassClassifier}]

# Có thể thêm loạt các multi-classifiers khác như BR, CC, 
classifier_config = [{"name":"CC - SVM", "classifier" : 'class CC_SVM vào chỗ này'}, 
			   {"name":"BR - SVM", "classifier" : 'class_SVM_vao_cho_nay'}
]

unlabel_config = [0, 100, 200, 300]
label_train_config = [500, 750, 1000]
label_test_config = [250]

