class Individual:
    def __init__(self, **kwargs):
        for attr in kwargs:
            self.__dict__[attr] = kwargs[attr]

    def copy(self):
        return Individual( **self.__dict__ )


    def __str__(self):
        return 'Ind({})'.format(self.__dict__)

    def __repr__(self):
        return repr('Ind({})'.format(self.__dict__))