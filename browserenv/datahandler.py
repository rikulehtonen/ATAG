import json
import os


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
        self.elements = []
        self.actions = []
        self.clickActions = ['A', 'BUTTON']
        self.typeActions = ['INPUT']
        self.ignoreElements = ['DIV']
        self.typeWordList = ['testaaja', 'testi', 'salasana']
        self.createFolders()

    def createFolders(self):
        path = self.folder + 'temp/'
        if not os.path.exists(path):
            os.makedirs(path)

    def __loadData(self, fileName):
        with open(fileName, 'r') as f:
            return json.load(f)

    def saveElements(self, elements):
        elementsFile = self.folder + 'temp/config_elements.json'
        if os.path.isfile(elementsFile):
            self.elements = self.__loadData(elementsFile)

        for e in elements:
            if e not in self.elements and e['tag'] not in self.ignoreElements:
                self.elements.append(e)

        with open(elementsFile, 'w') as f:
            json.dump(self.elements, f)

    def __appendToActions(self, action):
        if action != None and action not in self.actions:
            self.actions.append(action)

    def __xpathGeneration(self, element):
        # TODO: Get all attributes automaticly
        xpath = "xpath=//{}".format(element['tag'])
        attributes = element.get('attributes')
        for attribute in attributes:
            key = attribute.get('key')
            value = attribute.get('value')
            if "'" not in key and "'" not in value:
                xpath += "[@{}='{}']".format(key, value)

        if element['text'] != None:
            xpath += "[contains(text(),'{}')]".format(element['text'])

        return xpath

    def saveActions(self, elements):
        actionsFile = self.folder + 'temp/config_actions.json'
        if os.path.isfile(actionsFile):
            self.actions = self.__loadData(actionsFile)

        for element in elements:
            # Check click actions
            if element['tag'] in self.clickActions:
                xpath = self.__xpathGeneration(element)
                action = {"keyword": "click", "args": [xpath]}
                self.__appendToActions(action)
            
            # Check type actions
            action = None
            if element['tag'] in self.typeActions:
                xpath = self.__xpathGeneration(element)
                for word in self.typeWordList:
                    action = {"keyword": "type_text", "args": [xpath, word]}
                    self.__appendToActions(action)


        with open(actionsFile, 'w') as f:
            json.dump(self.actions, f)