import json
import os


class DataLoad:
    def __init__(self, config):
        self.config = config
        self.elements = None
        self.actions = None
        self.getFromFiles()

    def lenElements(self):
        return len(self.elements)
    
    def lenActions(self):
        return len(self.actions)
    
    def getFromFiles(self):
        conf_path = self.config.env_parameters.get('config_path')
        conf_elements = conf_path + self.config.env_parameters.get('elements_file')
        conf_actions = conf_path + self.config.env_parameters.get('actions_file')

        with open(conf_elements, 'r') as f:
            self.elements = json.load(f)

        with open(conf_actions, 'r') as f:
            self.actions = json.load(f)

    def get_action(self, index):
        return self.actions[index]


class DataSave:
    def __init__(self, config):
        self.config = config
        self.elements = []
        self.actions = []
        self.createFolders()

    def createFolders(self):  
        path = self.config.data_collection.get('temp_config_path')
        if not os.path.exists(path):
            os.makedirs(path)

    def __loadData(self, fileName):
        with open(fileName, 'r') as f:
            return json.load(f)

    def saveElements(self, elements):
        path = self.config.data_collection.get('temp_config_path')
        elementsFile = path + self.config.data_collection.get('elements_file')
        if os.path.isfile(elementsFile):
            self.elements = self.__loadData(elementsFile)

        for e in elements:
            ignoreElements = self.config.data_collection.get('ignore_elements')
            if e not in self.elements and e['tag'] not in ignoreElements:
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
        path = self.config.data_collection.get('temp_config_path')
        actionsFile = path + self.config.data_collection.get('actions_file')
        if os.path.isfile(actionsFile):
            self.actions = self.__loadData(actionsFile)

        for element in elements:
            # Check click actions
            if element['tag'] in self.config.data_collection.get('click_actions'):
                xpath = self.__xpathGeneration(element)
                action = {"keyword": "click", "args": [xpath]}
                self.__appendToActions(action)
            
            # Check type actions
            action = None
            if element['tag'] in self.config.data_collection.get('type_actions'):
                xpath = self.__xpathGeneration(element)
                for word in self.config.data_collection.get('type_word_list'):
                    action = {"keyword": "type_text", "args": [xpath, word]}
                    self.__appendToActions(action)

        with open(actionsFile, 'w') as f:
            json.dump(self.actions, f)