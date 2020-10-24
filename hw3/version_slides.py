from planning import PlanningProblem, Action, Expr, expr
import planning
import string

import clingo

def solve_planning_problem_using_ASP(planning_problem,t_max):
    """
    If there is a plan of length at most t_max that achieves the goals of a given planning problem,
    starting from the initial state in the planning problem, returns such a plan of minimal length.
    If no such plan exists of length at most t_max, returns None.

    Finding a shortest plan is done by encoding the problem into ASP, calling clingo to find an
    optimized answer set of the constructed logic program, and extracting a shortest plan from this
    optimized answer set.

    NOTE: still needs to be implemented. Currently returns None for every input.

    Parameters:
        planning_problem (PlanningProblem): Planning problem for which a shortest plan is to be found.
        t_max (int): The upper bound on the length of plans to consider.

    Returns:
        (list(Expr)): A list of expressions (each of which specifies a ground action) that composes
        a shortest plan for planning_problem (if some plan of length at most t_max exists),
        and None otherwise.
    """
    print("max t:", t_max)
    print("initial:", planning_problem.initial)
    print("goal:", planning_problem.goals)
    print("actions:{}\n".format(planning_problem.actions))
    for action in planning_problem.actions:
        print("action name:", action.name)
        print("action args:", action.args)
        print("action precon:", action.precond)
        print("action effect:", action.effect)

    print("\n")

    import clingo
    import string

    asp_code = ""

    # Define time constants
    asp_code += "#const t_max={}.\n".format(t_max)
    for t in range(t_max):
        asp_code += "time({}).\n".format(t)

    # Define fluents
    for action in planning_problem.actions:
        action_code, _ = get_action_code(action)

        for effect in action.effect:
            fluent_code = ""
            possible_code = ""

            effect_name = get_name(effect)
            effect_args = convert_args(effect)

            if effect_name[0] != "~":
                if effect_args:
                    fluent_code += "fluent({}(".format(effect_name)
                    prev_arg = []
                    for arg in effect_args:
                        fluent_code += """{}, """.format(arg)
                        if prev_arg:
                            for p in prev_arg:
                                possible_code += "{} != {}, ".format(p, arg)
                        prev_arg.append(arg)
                    fluent_code = fluent_code[:-2] + "))"
                else:
                    fluent_code += "fluent({})".format(effect_name)
                
                for precond in action.precond:
                    fluent_code += ":- "
                    precond_name = str(precond).split("(", 1)[0].lower()
                    precond_args = convert_args(precond)
                    if precond_args:
                        fluent_code += "{}(".format(precond_name)
                        for arg in precond_args:
                            fluent_code += """{}, """.format(arg)
                        fluent_code = fluent_code[:-2] + "), "
                    else:
                        fluent_code += "{}, ".format(precond_name)
                    fluent_code = fluent_code[:-2]
                fluent_code += possible_code
                fluent_code += ".\n"
                asp_code += fluent_code 

    asp_code += "literal(F):- fluent(F).\n"

    for state in planning_problem.initial:
        state_name = str(state).split("(", 1)[0].lower()
        state_args = convert_args(state)
        if state_args:
            asp_code += "{}(".format(state_name)
            for arg in state_args:
                asp_code += """{}, """.format(arg)
            asp_code = asp_code[:-2] + ").\n"
        else:
            asp_code += "{}.\n".format(state_name)

    # Define states
    for state in planning_problem.initial:
        asp_code += "state({}, 0).\n".format(str(state).lower())
    asp_code += "state(neg(F), 0):- fluent(F), not state(F, 0).\n"


    asp_code += "\n"
    
    # Define options
    for action in planning_problem.actions:
        action_code, possible_code = get_action_code(action)
        print(action)

        asp_code += "option({})".format(action_code)
        for precond in action.precond:
            asp_code += ":- "
            precond_name = str(precond).split("(", 1)[0].lower()
            precond_args = convert_args(precond)
            if precond_args:
                asp_code += "{}(".format(precond_name)
                for arg in precond_args:
                    asp_code += """{}, """.format(arg)
                asp_code = asp_code[:-2] + "), "
            else:
                asp_code += "{}, ".format(precond_name)
            asp_code = asp_code[:-2]
        asp_code += possible_code
        asp_code += ".\n"


    # Define states?
    # for action in planning_problem.actions:
    #     action_code, _ = get_action_code(action)

    #     for effect in action.effect:
    #         state_code = ""
    #         effect_name = get_name(effect)
    #         effect_args = convert_args(effect)
        
    #         if effect_name[0] != "~":
    #             if effect_args:
    #                 state_code += "state(neg({}(".format(effect_name)
    #                 for arg in effect_args:
    #                     state_code += """{}, """.format(arg)
    #                 state_code = state_code[:-2] + ")), T), "
    #             else:
    #                 state_code += "state(neg({}, T)), ".format(effect_name)
    #         state_code = state_code[:-2] + ":- "
    #         asp_code += state_code 

    #         for precond in action.precond:
    #             precond_name = str(precond).split("(", 1)[0].lower()
    #             precond_args = convert_args(precond)
    #             if precond_args:
    #                 asp_code += "{}(".format(precond_name)
    #                 for arg in precond_args:
    #                     asp_code += """{}, """.format(arg)
    #                 asp_code = asp_code[:-2] + "), "
    #             else:
    #                 asp_code += "{}, ".format(precond_name)
    #         asp_code += possible_code
    #         asp_code = asp_code[:-2] + ".\n"

    asp_code += "\n"

    # Define available actions at T
    for action in planning_problem.actions:
        action_code, possible_code = get_action_code(action)
        print(action)

        asp_code += "available({}, T):- ".format(action_code)
        for precond in action.precond:
            precond_name = str(precond).split("(", 1)[0].lower()
            precond_args = convert_args(precond)
            if precond_args:
                asp_code += "state({}(".format(precond_name)
                for arg in precond_args:
                    asp_code += """{}, """.format(arg)
                asp_code = asp_code[:-2] + "), T), "
            else:
                asp_code += "state({}, T), ".format(precond_name)

        # asp_code += "not contradict({}, T), time(T).\n".format(action_code)

        # asp_code += "contradict({}, T):- ".format(action_code)
        for effect in action.effect:
            effect_name = str(effect).split("(", 1)[0].lower()
            effect_args = convert_args(effect)
            if effect_name[0] != "~":
                if effect_args:
                    asp_code += "state(neg({}(".format(effect_name)
                    for arg in effect_args:
                        asp_code += """{}, """.format(arg)
                    asp_code = asp_code[:-2] + ")), T), "
                else:
                    asp_code += "state(neg({}, T)), ".format(effect_name)
        # asp_code += "not action({}, T), ".format(action_code)
        # asp_code = asp_code[:-2] + ".\n"
        asp_code += possible_code
        asp_code += "time(T).\n"

    # asp_code += """available(A, T):- not not_available(A, T).\n"""
    # asp_code += """not_available(A, T):- not available(A, T).\n"""

    # asp_code += """action(A, T):- blocked(A, T).\n"""
    # asp_code += ":- "

    asp_code += "\n"

    # Constraint on action
    asp_code += ":- option(A), time(T), action(A, T), not available(A, T).\n"

    asp_code += "\n"


    # Represent action effect
    for action in planning_problem.actions:
        action_code, _ = get_action_code(action)
        
        for effect in action.effect:
            state_code = ""
            effect_name = get_name(effect)
            effect_args = convert_args(effect)
        
            if effect_name[0] != "~":
                if effect_args:
                    state_code += "state({}(".format(effect_name)
                    for arg in effect_args:
                        state_code += """{}, """.format(arg)
                    state_code = state_code[:-2] + "), T+1), "
                else:
                    state_code += "state({}, T+1), ".format(effect_name)
            else:
                effect_name = effect_name[1:]
                if effect_args:
                    state_code += "state(neg({}(".format(effect_name)
                    for arg in effect_args:
                        state_code += """{}, """.format(arg)
                    state_code = state_code[:-2] + ")), T+1), "
                else:
                    state_code += "state(neg({}, T+1)), ".format(effect_name)
            state_code = state_code[:-2] 
            state_code += ":- "
            asp_code += state_code 

            for precond in action.precond:
                precond_name = str(precond).split("(", 1)[0].lower()
                precond_args = convert_args(precond)
                if precond_args:
                    asp_code += "{}(".format(precond_name)
                    for arg in precond_args:
                        asp_code += """{}, """.format(arg)
                    asp_code = asp_code[:-2] + "), "
                else:
                    asp_code += "{}, ".format(precond_name)
            asp_code += possible_code 
            asp_code += "action({}, T).\n".format(action_code)

    asp_code += "\n"

    # The inertial rule
    for action in planning_problem.actions:
        action_code, _ = get_action_code(action)
        
        for effect in action.effect:
            state_code = ""
            effect_name = get_name(effect)
            effect_args = convert_args(effect)
        
            if effect_name[0] != "~":
                if effect_args:
                    state_code += "state({}(".format(effect_name)
                    for arg in effect_args:
                        state_code += """{}, """.format(arg)
                    state_code = state_code[:-2] + "), T+1), "
                else:
                    state_code += "state({}, T+1), ".format(effect_name)

                state_code = state_code[:-2] 
                state_code += ":- "
                asp_code += state_code 

                for precond in action.precond:
                    precond_name = str(precond).split("(", 1)[0].lower()
                    precond_args = convert_args(precond)
                    if precond_args:
                        asp_code += "{}(".format(precond_name)
                        for arg in precond_args:
                            asp_code += """{}, """.format(arg)
                        asp_code = asp_code[:-2] + "), "
                    else:
                        asp_code += "{}, ".format(precond_name)
                asp_code += possible_code 


                state_code = ""
                if effect_args:
                    state_code += "state({}(".format(effect_name)
                    for arg in effect_args:
                        state_code += """{}, """.format(arg)
                    state_code = state_code[:-2] + "), T), "
                else:
                    state_code += "state({}, T), ".format(effect_name)

                if effect_args:
                    state_code += "not state(neg({}(".format(effect_name)
                    for arg in effect_args:
                        state_code += """{}, """.format(arg)
                    state_code = state_code[:-2] + ")), T+1), "
                else:
                    state_code += "not state(neg({}, T+1)), ".format(effect_name)

 
                asp_code += state_code 

                asp_code += "time(T).\n"


        # asp_code = asp_code[:-2]
        # asp_code += ":- action({}, T).\n".format(action_code, action_code)
        # asp_code += "action({}, T+1){}:- available({}, T), time(T), T<t_max.\n".format(action_code, "}", action_code)
        # asp_code += "action({}, T+1):- available({}, T), time(T), T<t_max.\n".format(action_code, action_code)

    # asp_code += """state(S, T+1):- T), not change_state(S, T), time(T).\n"""

        # asp_code += ":- action({}, T).\n".format(action_code, action_code)
        # asp_code += ":- time(T), T<t_max.\n".format(action_code, "}", action_code)
        # asp_code += "action({}, T+1):- available({}, T), time(T), T<t_max.\n".format(action_code, action_code)
    
    # asp_code += "{ action(A, T) : time(T) }."
    # Inertia
    # state_code = """state(S, T+1):- state(S, T), not change_state(S, T), time(T).\n"""
    


    # for action in planning_problem.actions:
    #     neg_state = False
    #     action_code, _ = get_action_code(action)
    #     state_code = ""
    #     for effect in action.effect:
    #         change_state = False
    #         effect_name = get_name(effect)
    #         effect_args = convert_args(effect)
        
    #         if effect_name[0] == "~":
    #             if effect_args:
    #                 effect_name = effect_name[1:]
    #                 state_code += "change_state({}(".format(effect_name)
    #                 for arg in effect_args:
    #                     state_code += """{}, """.format(arg)
    #                 state_code = state_code[:-2] + "), T), "
    #             else:
    #                 state_code += "change_state({}, T), ".format(effect_name)
    #             change_state = True
    #             neg_state = True
    #     if change_state:
    #         state_code = state_code[:-2]
    #         state_code += ":- action({}, T).\n".format(action_code)
    # if not neg_state:
    #     state_code = "time(T).\n"
    # asp_code += state_code



    asp_code += "\n"

    
    asp_code += """goal(T): """
    for goal in planning_problem.goals:
        goal_name = str(goal).split("(", 1)[0].lower()
        goal_args = convert_args(goal)
        if goal_args:
            asp_code += "state({}(".format(goal_name)
            for arg in goal_args:
                asp_code += """{}, """.format(arg)
            asp_code = asp_code[:-2] + "), T), "
        else:
            asp_code +=  "state({}, T), ".format(goal_name)
    asp_code = asp_code[:-2] 
    asp_code += ".\n"
    asp_code += "goal(T+1):- time(T), goal(T).\n"

    asp_code += "1 { action(A, T) : available(A, T) } 1:- time(T), T<t_max, not goal(T).\n"
    asp_code += """#minimize { T: action(_, T) }.\n"""
    asp_code += """#show action/2."""
    # asp_code += """#show available/2."""

    asp_code += "\n"

    print("ASP CODE:\n")
    print(asp_code)

    output = get_optimal_answer_sets(asp_code)
    print("output:", output)
    new_output = dict()
    for action in output:
        action = action.split("(", 1)[1][:-1]
        number = action[-1:]
        action = action[:-2]
        new_output[number] = action
    new_output = [new_output[k] for k in sorted(new_output)]
    return new_output

def get_name(instance):
    return str(instance).split("(", 1)[0].lower()

def get_action_code(action):
    action_code = ""
    args = convert_args(action.args)
    possible_code = ""
    if args:
        action_code += """{}(""".format(action.name.lower())
        prev_arg = []
        for arg in args:
            action_code += """{}, """.format(arg)
            if prev_arg:
                for p in prev_arg:
                    possible_code += "{} != {}, ".format(p, arg)
            prev_arg.append(arg)
        action_code = action_code[:-2] + ")"
    else:
        action_code += """{}""".format(action.name.lower())
    return action_code, possible_code

def convert_args(args):
    if str(args) == "()" or "(" not in str(args):
        return []
    args = str(args).split("(", 1)[1][:-1].split(", ")
    new_args = []
    for arg in args:
        new_arg = list(arg)
        if new_arg[0].isupper():
            new_arg[0] = new_arg[0].lower()
        elif new_arg[0].islower():
            new_arg[0] = new_arg[0].upper()
        new_arg = "".join(new_arg)
        new_args.append(new_arg)
    args = new_args
    return args

def get_optimal_answer_sets(program):
    # Load the answer set program, and call the grounder
    control = clingo.Control();
    control.add("base", [], program);
    control.ground([("base", [])]);
    # Define a function that will be called when an answer set is found
    # This function sorts the answer set alphabetically, and prints it
    def on_model(model):
        global output
        if model.optimality_proven == True:
            sorted_model = [str(atom) for atom in model.symbols(shown=True)];
            sorted_model.sort();
            output = sorted_model
        
        for atom in model.symbols(atoms=True):
            print(model.symbols(shown=True));
   

    # Ask clingo to find all optimal models (using an upper bound of 0 gives all models)
    control.configuration.solve.opt_mode = "optN";
    control.configuration.solve.models = 1;

    answer = control.solve(on_model=on_model)

    if answer.satisfiable:
        print("satisfiable")
    else:
        print("unsatisfiable")

    return output