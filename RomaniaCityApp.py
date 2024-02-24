"""
Azzam Shaikh
CS 534 - Artificial Intelligence
Individual Project # 2
March 3, 2024

This script contains the code required to run the Romania City App. Portions of the code has been extracted from
the textbook source code (https://github.com/aimacode/aima-python).  All code used from the source code will be
acknowledged as per the Honor Code.
"""

from SimpleProblemSolvingAgent import *


class RomaniaCityApp:
    """
    This RomaniaCityApp class is the main class to create the application.
    """
    def __init__(self):
        """
        Upon initialization, this class stores the Romania map, Romania map locations,
        initial and goal states, and supporting variables to run the program.
        """
        self.romania_map = UndirectedGraph(dict(
                            Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
                            Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
                            Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
                            Drobeta=dict(Mehadia=75),
                            Eforie=dict(Hirsova=86),
                            Fagaras=dict(Sibiu=99),
                            Hirsova=dict(Urziceni=98),
                            Iasi=dict(Vaslui=92, Neamt=87),
                            Lugoj=dict(Timisoara=111, Mehadia=70),
                            Oradea=dict(Zerind=71, Sibiu=151),
                            Pitesti=dict(Rimnicu=97),
                            Rimnicu=dict(Sibiu=80),
                            Urziceni=dict(Vaslui=142)))

        self.romania_map.locations = dict(
                            Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
                            Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
                            Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
                            Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
                            Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
                            Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
                            Vaslui=(509, 444), Zerind=(108, 531))

        self.initial = None
        self.goal = None
        self.run = True
        self.iteration = 1

    def __call__(self):
        while self.run:
            self.program_start(self.iteration)
            spsa = SimpleProblemSolvingAgentProgram(self.initial, self.romania_map,
                                                    self.romania_map.locations)
            spsa(self.goal)
            self.program_complete()
            self.iteration += 1

    @staticmethod
    def initial_starting_query():
        """
        Method for getting user initial starting city
        """
        return str(input('\nPlease enter the origin city: ')).capitalize()

    @staticmethod
    def second_starting_query(city):
        """
        Second method for getting user initial starting city if the first user input was not valid
        """
        return str(input('\nCould not find ' + city + ', please try again: ')).capitalize()

    @staticmethod
    def initial_destination_query():
        """
        Method for getting user initial destination city
        """
        return str(input('\nPlease enter the destination city: ')).capitalize()

    @staticmethod
    def second_destination_query(city):
        """
        Second method for getting user destination starting city if the first user input was not valid
        """
        return str(input('\nCould not find ' + city + ', please try again: ')).capitalize()

    def get_origin_city(self, start_city):
        """
        A recursive function to get the origin city. This function ensures
        the start city is a valid city. Otherwise, it will continue to ask
        the user for a valid city.
        """
        if start_city in self.romania_map.locations.keys():
            return start_city
        else:
            start_city = self.second_starting_query(start_city)
            return self.get_origin_city(start_city)

    def get_destination_city(self, goal_city):
        """
        A recursive function to get the destination city. This function ensures
        the destination city is a valid city. Otherwise, it will continue to ask
        the user for a valid city.
        """
        if goal_city in self.romania_map.locations.keys():
            return goal_city
        else:
            goal_city = self.second_destination_query(goal_city)
            return self.get_destination_city(goal_city)

    def get_user_input(self):
        """
        A function which combines the previous functions to get the user input
        """
        while True:
            start_city = self.initial_starting_query()
            start_city = self.get_origin_city(start_city)
            goal_city = self.initial_destination_query()
            if goal_city == start_city:
                print('\nThe same city can\'t be both origin and destination. Please try again.')
                continue
            goal_city = self.get_destination_city(goal_city)
            if goal_city != start_city:
                return start_city, goal_city
            else:
                print('\nThe same city can\'t be both origin and destination. Please try again.')
                continue

    def program_start(self, ii):
        """
        A function to be executed at the start of the program. It contains the
        functions to get the user input.
        """
        if ii == 1:
            print('Here are all the possible Romania cities that can be traveled:\n')
            print(list(self.romania_map.locations.keys()))
        self.initial, self.goal = self.get_user_input()

    def program_complete(self):
        """
        A function to be executed at the end of the program. It contains the
        functions get a user input about continuing the program.
        """
        response = str(input('\nWould you like to find the best path between other two cities? ')).capitalize()
        if response == 'Yes':
            self.run = True
        elif response == 'No':
            print('\nThank You for Using Our App')
            self.run = False
        else:
            print('\nThank You for Using Our App')
            self.run = False


def main():
    """
    This main function runs the RomaniaCityApp.
    """
    romania_app = RomaniaCityApp()     # Create a RomaniaCityApp object
    romania_app()                      # Call the object to run the program


if __name__ == "__main__":
    main()
