

class DataSave:
    def __init__(self, folder):
        self.folder = folder

    def elements(self):
        pass

    def actions(self):
        pass

    def targets(self):
        pass



class DataLoad:
    def __init__(self, folder):
        self.folder = folder
        self.elements = None
        self.actions = None
        self.targets = None

    def elements(self):
        return self.elements

    def action(self, index):
        return self.actions[index]

    def targets(self):
        return(self.targets)