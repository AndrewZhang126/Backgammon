'''
Name(s): Andrew Zhang, Alex Bergstrom
UW netid(s): azhang26, bergalex

Modified strater code for DSBG with minimax and alpha-beta pruning

Assignment 3, CSE 415, University of Washington
'''

from game_engine import genmoves

class BackgammonPlayer:
    def __init__(self):
        self.GenMoveInstance = genmoves.GenMoves()
        self.ply_value = 2
        self.own_static = True
        self.given_static = None
        self.use_prune = False
        self.state_counter = 0
        self.cutoff_counter = 0
        # feel free to create more instance variables as needed.

    #return a string containing your UW NETID(s)
    # For students in partnership: UWNETID + " " + UWNETID
    def nickname(self):
        return "2035431" + " " + "2050354"

    # If prune==True, then your Move method should use Alpha-Beta Pruning
    # otherwise Minimax
    def useAlphaBetaPruning(self, prune=False):
        self.use_prune = prune
        self.state_counter = 0
        self.cutoff_counter = 0

    # Returns a tuple containing the number explored
    # states as well as the number of cutoffs.
    def statesAndCutoffsCounts(self):
        return (self.state_counter, self.cutoff_counter)

    # Given a ply, it sets a maximum for how far an agent
    # should go down in the search tree. maxply=2 indicates that
    # our search level will go two level deep.
    def setMaxPly(self, maxply=2):
        self.ply_value = maxply

    # If not None, it update the internal static evaluation
    # function to be func
    def useSpecialStaticEval(self, func):
        if not func == None:
            self.own_static = False
            self.given_static = func

    # Given a state and a roll of dice, it returns the best move for
    # the state.whose_move.
    # Keep in mind: a player can only pass if the player cannot move any checker with that role
    def move(self, state, die1=1, die2=6):
        best_move = 'p'
        initial_value_W = -100000
        initial_value_R = 100000
        self.initialize_move_gen_for_state(state, state.whose_move, die1, die2)
        moves = self.get_all_possible_moves()
        #if no possible moves can be made, then pass
        if len(moves) == 0:
            return best_move
        for move in moves:
            self.state_counter += 1
            if move != 'p':
                if state.whose_move == 0:
                    if not self.use_prune:
                        current_value = self.minimax(move[1], 1, self.ply_value)
                    else:
                        current_value = self.alpha_beta(move[1], 1, self.ply_value, -100000, 100000)
                    #maximizing agent will choose state with largest value
                    if current_value > initial_value_W:
                        best_move = move[0]
                        initial_value_W = current_value
                if state.whose_move == 1:
                    if not self.use_prune:
                        current_value = self.minimax(move[1], 0, self.ply_value)
                    else:
                        current_value = self.alpha_beta(move[1], 0, self.ply_value, -100000, 100000)
                    #minimizing agent will choose state with smallest value
                    if current_value < initial_value_R:
                        best_move = move[0]
                        initial_value_R = current_value  
        return best_move

    def initialize_move_gen_for_state(self, state, who, die1, die2):
        self.move_generator = self.GenMoveInstance.gen_moves(state, who, die1, die2)

    #apply minimax search for a given state
    def minimax(self, state, whoseMove, ply):
        #when you reach the end of searching the tree, calculate the static eval function on those nodes
        if ply == 0:
            if self.own_static:
                return self.staticEval(state)
            else:
                return self.given_static(state)
        #white is maximizing agent
        if whoseMove == 0:
            provisional = -100000
        elif whoseMove == 1:
            provisional = 100000
        #get each move possible from current state
        self.initialize_move_gen_for_state(state, whoseMove, 1, 6)
        for move in self.get_all_possible_moves():
            self.state_counter += 1
            if move != 'p':
                #keep traversing tree
                new_value = self.minimax(move[1], 1 - whoseMove, ply - 1)
                #update provisional value if maximizing or minimizing agent has better option
                if (whoseMove == 0 and new_value > provisional) or (whoseMove == 1 and new_value < provisional):
                    provisional = new_value  
        return provisional


    #alpha beta pruning needs an interval [alpha, beta] that determines where solution must lie
    def alpha_beta(self, state, whoseMove, ply, alpha, beta):
        #when you reach the end of searching the tree, calculate the static eval function on those nodes
        if ply == 0:
            if self.own_static:
                return self.staticEval(state)
            else:
                return self.given_static(state)
        #white is maximizing agent
        if whoseMove == 0:
            provisional = -100000
            self.initialize_move_gen_for_state(state, whoseMove, 1, 6)
            for move in self.get_all_possible_moves():
                self.state_counter += 1
                if move != 'p':
                    #alpha value may change after results of minimizing agent, so pass in provisional
                    #as argument instead of alpha
                    new_value = self.alpha_beta(move[1], 1 - whoseMove, ply - 1, provisional, beta)
                    #alpha cutoff occurs at maximing node when maximizing agent has move that
                    #results in values greater than beta, or outside the solution range
                    #since minimizing agent will not consider those values, there is no point exploring further
                    if provisional > beta:
                        self.cutoff_counter += 1
                        return beta
                    if new_value > provisional:
                        provisional = new_value
            return provisional
        if whoseMove == 1:
            provisional = 100000
            self.initialize_move_gen_for_state(state, whoseMove, 1, 6)
            for move in self.get_all_possible_moves():
                self.state_counter += 1
                if move != 'p':
                    #beta value may change after results of minimizing agent, so pass in provisional
                    #as argument instead of beta
                    new_value = self.alpha_beta(move[1], 1 - whoseMove, ply - 1, alpha, provisional)
                    #alpha cutoff occurs at minimizing node when minimizing agent has move that
                    #results in values less than alpha, or outside the solution range
                    #since maximizing agent will not consider those values, there is no point exploring further
                    if provisional < alpha:
                        self.cutoff_counter += 1
                        return alpha
                    if new_value < provisional:
                        provisional = new_value
            return provisional
    
    def get_all_possible_moves(self):
        """Uses the mover to generate all legal moves. Returns an array of move commands"""
        move_list = []
        done_finding_moves = False
        any_non_pass_moves = False
        while not done_finding_moves:
            try:
                m = next(self.move_generator)    # Gets a (move, state) pair.
                if m[0] != 'p':
                    any_non_pass_moves = True
                    move_list.append(m)    # Add the (move, state) to the list.
            except StopIteration as e:
                done_finding_moves = True
        if not any_non_pass_moves:
            move_list.append('p')
        return move_list


    def staticEval(self, state):
        '''more positive means better for white, more negative means better for red
        Red to Bear Off < White to Bear Off < White Hit From Bar < Red Hit From Bar < White About to Win
        -red to bear off means all checkers are now in red's home base, so
        only pointLists[19] through pointLists[24] have only red checkers
        -white to bear off means the same as red to bear off, but only pointLists[0] through pointLists[5]
        have white checkers
        -white and red hit from bar are when bearing off is occuring
        -white hit from bar means you have gotten checkers off the board from bearing off, but one of white's
        checkers has been hit so the bear off process has been stopped until the checker canr return home.
        this is worse than red hit from bar because you have been hit and cannot bear off temporarily
        -white about to win means self.white_off is almost 12 and all white checkers are in home and red
        cannot hit white
        can bear off < have checkers off board
        '''
        white_home = 0
        white_other_home = 0
        white_bear_off = 0
        red_home = 0
        red_other_home = 0
        red_bear_off = 0
        #counts the checkers in each players hpme
        for i in range(0, 6):
            for wc in state.pointLists[i]:
                if wc == 0:
                    white_home += 1
                else:
                    #red in white's home is good as it prevents white bearing off
                    red_other_home +=1
            for rc in state.pointLists[i+18]:
                if rc == 1:
                    red_home += 1
                else:
                    white_other_home += 1
        white_not_home = 15 - len(state.white_off) - white_home
        red_not_home = 15 - len(state.red_off) - red_home
        #determines if either white or red can bear off by checking if all checkers are home or finished
        #if a player can bear off, that is good for them
        if white_not_home == red_other_home == 0:
            white_bear_off += 30
        if red_not_home == 0 == white_other_home == 0:
            red_bear_off += 30
        return (red_not_home + 2*white_other_home + white_bear_off + 25*len(state.white_off)) - (white_not_home + 2*red_other_home + red_bear_off + 25*len(state.red_off))
