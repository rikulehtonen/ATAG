import json
import os
import numpy as np
import base64
import json


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


class PathSave:
    def __init__(self, config):
        self.config = config
        self.depth = 0
        self.prevstate = None
        self.path = []

    def reset(self):
        self.depth = 0
        self.prevstate = None

    def checkDepth(self):
        if not (len(self.path) > self.depth):
            self.path.append({})

    def saveToFile(self):
        file_path = self.config.data_collection.get('temp_config_path') + self.config.data_collection.get('collect_path_file')
        with open(file_path, "w") as json_file:
            json.dump(self.path, json_file)

    def save(self, obs, done, label):
        obs = np.packbits(obs)
        state = bytearray(obs)
        state = base64.b64encode(state).decode('utf-8')
        if state == self.prevstate:
            return False

        self.checkDepth()
        layer = self.path[self.depth]
        connections = layer.get(self.prevstate)

        if connections == None:
            connections = {state: {'visits': 1, 'done': done, 'label': label}}
        elif state in connections.keys():
            connections[state]['visits'] += 1
            connections[state]['done'] = connections[state]['done'] or done
            connections[state]['label'] = label
        else:
            connections.update({state: {'visits': 1, 'done': done, 'label': label}})

        layer.update({self.prevstate: connections})
        self.path[self.depth] = layer

        if self.prevstate == None:
            self.saveToFile()

        self.prevstate = state
        self.depth += 1
        return True
    

class TrainingData:
    def __init__(self, config):
        self.config = config
        self.dataitems = []

    def save(self,ep_obs,ep_next_obs,ep_actions,ep_act_probs,ep_rewards,ep_dones):
        # Create the directory if it doesn't exist
        os.makedirs(self.config['training_data_path'], exist_ok=True)
        
        # Create a dictionary to store your training data
        training_data = {
            "observations": np.asarray(ep_obs).tolist(),
            "next_observations": np.asarray(ep_next_obs).tolist(),
            "actions": np.asarray(ep_actions).tolist(),
            "act_probs": np.asarray(ep_act_probs).tolist(),
            "rewards": np.asarray(ep_rewards).tolist(),
            "terminals": np.asarray(ep_dones).tolist(),
        }
        
        self.dataitems.append(training_data)

        # Generate a unique filename for each training session based on timestamp
        filename = os.path.join(self.config['training_data_path'], self.config['filename'])
        
        # Save the dictionary as a JSON file
        with open(filename, 'w') as f:
            json.dump(self.dataitems, f)

    