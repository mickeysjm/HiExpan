"""
__author__: Ellen Wu, Jiaming Shen
__description__: A utility class representing TreeNode
"""


class TreeNode:

    def __init__(self, parent=None, level=-1, eid=-1, ename="NONE", isUserProvided=False, confidence_score=None,
                 max_children=100000000):
        """

        :param parent: a single TreeNode object
        :param level: zero indexed level (int), -1 means the root
        :param eid: int
        :param ename: string
        :param isUserProvided: True/False
        """
        self.parent = parent
        self.children = []
        self.level = level
        self.eid = eid
        self.ename = ename
        self.restrictions = set()    # a set of eids
        self.isUserProvided = isUserProvided
        self.synonyms = set()
        self.confidence_score = confidence_score
        self.max_children = max_children    # whether to further expand children under this node

    def __str__(self):
        if not self.parent:
            return "RootNode"
        else:
            return ("%s (eid=%s,log_prob=%0.6f,parent=%s)" %
                    (self.ename, self.eid, self.confidence_score, self.parent.ename)
                    )

    def addRestriction(self, eid):
        if self.children == None:
            raise Exception("[WARNING] Unable to add restriction in empty TreeNode object")
        self.restrictions.add(self, eid)

    def addSynonym(self, eid):
        self.synonyms.add(eid)

    def setConfidenceScore(self, confidence_score):
        self.confidence_score = confidence_score

    def addChildren(self, children):
        if self.children == None:
            raise Exception("[WARNING] Unable to add children in empty TreeNode object")
        self.children += children

    def cutFromChild(self, child):
        """

        :param child: a TreeNode object
        :return:
        """
        if self.children == None:
            return
        self.restrictions.add(child.eid)
        index = None
        for i in range(len(self.children)):
            if self.children[i].eid == child.eid:
                index = i
                break
        self.children = self.children[:index]

    def updateFromChild(self, child):
        if self.children == None:
                raise Exception("[WARNING] Unable to update from child in empty TreeNode object")
        self.cutFromChild(child)
        self.addRestriction(child.eid)

    def isQualifiedChild(self, eid):
        return (eid in self.restrictions)

    def printSubtree(self, tabs):
        for i in range(tabs):
            print('\t', end="")
        print(self.ename, " (eid=%s)" % (self.eid))
        for child in self.children:
            child.printSubtree(tabs+1)

    def delete(self):
        self.parent = None
        self.children = None
        self.restrictions = None

    def saveToFile(self, outputFilePath):
        with open(outputFilePath,"w") as fout:
            s = []
            s.append(self)
            while(len(s) != 0):
                top = s[-1]
                s = s[:-1]
                tab_number = top.level + 1
                entity_name = top.ename
                fout.write(tab_number * "\t" + entity_name + " (eid=%s)" % (top.eid) + "\n")
                for child in reversed(top.children):
                    s.append(child)