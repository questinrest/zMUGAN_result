"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
from kegnet.classifier.train import main as train_student
from kegnet.generator.train import main as train_generator


def main():
    
    dataset = 'fashion'
    n_generators = 5
    path_teacher = f'../pretrained/{dataset}.pth.tar'
    path_out = f'../out/{dataset}'

    generators = []
    for i in range(n_generators):
        path_gen = f'{path_out}/generator-{i}'
        path_model = train_generator(dataset, path_teacher, path_gen, i)
        generators.append(path_model)

    # data_dist = 'real'
    # path_cls = f'{path_out}/classifier-{seed}'
    # option = 1
    # seed = 0
    # train_student(dataset, data_dist, path_cls, seed)


if __name__ == '__main__':
    main()
