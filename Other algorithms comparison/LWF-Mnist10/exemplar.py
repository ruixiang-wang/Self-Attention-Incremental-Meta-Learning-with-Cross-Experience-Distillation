class Exemplar:
    def __init__(self, max_size, total_cls):
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size
        self.total_classes = total_cls

    def update(self, cls_num, train):
        train_x, train_y = train
        assert self.cur_cls == len(list(self.train.keys()))
        
        self.cur_cls += 10
        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else max_size
        train_store_num = int(total_store_num )
        for key, value in self.train.items():
            self.train[key] = value[:train_store_num]
        
        for x, y in zip(train_x, train_y):
            if y not in self.train:
                self.train[y] = [x]
            else:
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)
        print(self.cur_cls)
        print(len(list(self.train.keys())))
        assert self.cur_cls == len(list(self.train.keys()))
        for key, value in self.train.items():
            assert len(self.train[key]) == train_store_num

    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_cur_cls(self):
        return self.cur_cls
