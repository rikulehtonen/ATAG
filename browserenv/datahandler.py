import json


class DataLoad:
    def __init__(self, folder):
        self.folder = folder
        self.elements = None
        self.actions = None
        self.targets = None

        self.getFromFiles()


    def lenElements(self):
        return len(self.elements)
    
    def lenActions(self):
        return len(self.actions)

    def getFromFiles(self):
        with open(self.folder + 'config_elements.json', 'r') as f:
            self.elements = json.load(f)

        with open(self.folder + 'config_actions.json', 'r') as f:
            self.actions = json.load(f)

        with open(self.folder + 'config_targets.json', 'r') as f:
            self.targets = json.load(f)

    def get_action(self, index):
        return self.actions[index]



class DataSave:
    def __init__(self, folder):
        self.folder = folder

    def elements(self):
        pass

    def actions(self):
        pass

    def targets(self):
        pass