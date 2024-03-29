from planning import PlanningProblem, Action, Expr, expr
import planning
import string
import numpy as np
import itertools

import clingo


def solve_planning_problem_using_ASP(planning_problem, t_max):
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
    import clingo
    import string

    asp_code = ""

    # Get objects
    objects = []
    objects_original_case = []

    # Dictionary that stores format of objects in ASP to original formatting
    case_dict = dict()

    # We go through all actions and states to obtain all objects
    for state in planning_problem.initial:
        name, args, var, _, _ = get_info(state)
        if var:
            var = np.invert(np.array(var))
            objects.append(np.array(args)[var].tolist())
        name, args, var, _, _ = get_info(state, convert_case=False)
        objects_original_case.append(args)

    for action in planning_problem.actions:
        name, args, var, _, _ = get_info(action)
        if var:
            var = np.invert(np.array(var))
            objects.append(np.array(args)[var].tolist())
        name, args, var, _, _ = get_info(action, convert_case=False)
        objects_original_case.append(args)

        for precond in action.precond:
            name, args, var, _, _ = get_info(precond)
            if var:
                var = np.invert(np.array(var))
                objects.append(np.array(args)[var].tolist())
            name, args, var, _, _ = get_info(precond, convert_case=False)
            objects_original_case.append(args)

        for effect in action.effect:
            name, args, var, _, _ = get_info(effect)
            if var:
                var = np.invert(np.array(var))
                objects.append(np.array(args)[var].tolist())

            name, args, var, _, _ = get_info(effect, convert_case=False)
            objects_original_case.append(args)
    for goal in planning_problem.goals:
        name, args, var, _, _ = get_info(goal)
        if var:
            var = np.invert(np.array(var))
            objects.append(np.array(args)[var].tolist())

        name, args, var, _, _ = get_info(goal, convert_case=False)
        objects_original_case.append(args)

    objects = list(set([item for sublist in objects for item in sublist]))
    objects_orig_case = list(
        set([item for sublist in objects_original_case for item in sublist]))

    # Construct unique case dict; format of objects in ASP to original formatting
    for o1 in objects:
        for o2 in objects_orig_case:
            if o1.lower() == o2.lower():
                if o2[0].isupper() and o1[0].islower():
                    case_dict[o1] = o2

    # Define time constants
    asp_code += "#const t_max={}.\n".format(t_max)
    for t in range(t_max+1):
        asp_code += "time({}).\n".format(t)

    # Define states at initial state
    for state in planning_problem.initial:
        asp_code += "state({}, 0).\n".format(str(state).lower())

    asp_code += "\n"

    # Define available actions at T
    for action in planning_problem.actions:
        action_code, possible_code = get_action_code(action)

        asp_code += "available({}, T):- ".format(action_code)

        # Action is available if all preconditions are met
        for precond in action.precond:
            name, args, var, _, _ = get_info(precond)
            if name[0] != "~":
                fluent_code = get_predicate_code("state", name, args, T="T")
            else:
                fluent_code = get_predicate_code(
                    "state", name[1:], args, T="T", neg=True)
            asp_code += fluent_code + ", "

        # If we already have all effects; make action not available.
        num_effects = len(action.effect)
        if num_effects > 0:
            asp_code += "not {} ".format(num_effects)
            asp_code += "{ "
        for effect in action.effect:
            name, args, var, _, _ = get_info(effect)
            if name[0] != "~":
                fluent_code = get_predicate_code("state", name, args, T="T")
            else:
                fluent_code = get_predicate_code(
                    "state", name[1:], args, T="T", neg=True)
            asp_code += fluent_code + "; "

        asp_code = asp_code[:-2]
        if num_effects > 0:

            asp_code += " }"
        asp_code += ", time(T).\n"

    asp_code += "\n"

    # Constraint on action
    asp_code += ":- time(T), action(A, T), not available(A, T).\n"

    asp_code += "\n"

    # Represent action effects: state is changed if it is an effect of an actua;
    # action at that timestep
    for action in planning_problem.actions:
        action_code, possible_code = get_action_code(action)

        for effect in action.effect:
            name, args, var, _, _ = get_info(effect)

            if name[0] != "~":
                state_code = get_predicate_code("state", name, args, T="T+1")
            else:
                state_code = get_predicate_code(
                    "state", name[1:], args, T="T+1", neg=True)

            state_code += ":- "
            asp_code += state_code

            asp_code += "action({}, T)".format(action_code)
            asp_code += ", time(T).\n"

    asp_code += "\n"

    # The inertial rules; states that hold for the next state if nothing happens
    # to these states. We look up states in effects and in preconditions
    for action in planning_problem.actions:
        action_code, possible_code = get_action_code(action)

        for effect in action.effect:
            name, args, var, _, _ = get_info(effect)

            if name[0] != "~":

                # For all positive states: state can transfer to next state if
                # we had the state already at previous timestep and we do not
                # have the negation in the next time step (as effect from action)
                state_code = get_predicate_code("state", name, args, T="T+1")

                state_code += ":- "
                asp_code += state_code

                state_code = get_predicate_code("state", name, args, T="T")
                state_code += ", "

                state_code += get_predicate_code("not state",
                                                 name, args, T="T+1", neg=True)

                state_code += ", time(T).\n"
                asp_code += state_code

                # For all states, but then the negations: same but vice versa
                # state remains negated when it was negated at previous timestep
                # and have not the not negated state
                state_code = get_predicate_code(
                    "state", name, args, T="T+1", neg=True)

                state_code += ":- "
                asp_code += state_code

                state_code = get_predicate_code(
                    "state", name, args, T="T", neg=True)
                state_code += ", "

                state_code += get_predicate_code("not state",
                                                 name, args, T="T+1")

                asp_code += state_code

                asp_code += ", time(T).\n"

        for precond in action.precond:

            name, args, var, _, _ = get_info(precond)

            if name[0] != "~":

                # For all positive states: state can transfer to next state if
                # we had the state already at previous timestep and we do not
                # have the negation in the next time step (as effect from action)
                state_code = get_predicate_code("state", name, args, T="T+1")

                state_code += ":- "
                asp_code += state_code

                state_code = get_predicate_code("state", name, args, T="T")
                state_code += ", "

                state_code += get_predicate_code("not state",
                                                 name, args, T="T+1", neg=True)

                state_code += ", time(T).\n"
                asp_code += state_code

                # For all states, but then the negations: same but vice versa
                # state remains negated when it was negated at previous timestep
                # and have not the not negated state
                state_code = get_predicate_code(
                    "state", name, args, T="T+1", neg=True)

                state_code += ":- "
                asp_code += state_code

                state_code = get_predicate_code(
                    "state", name, args, T="T", neg=True)
                state_code += ", "

                state_code += get_predicate_code("not state",
                                                 name, args, T="T+1")

                asp_code += state_code

                asp_code += ", time(T).\n"

    asp_code += "\n"

    # Formulate goal as a binary state with conditions being the goal states
    asp_code += """goal(T):- """
    for goal in planning_problem.goals:
        name, args, var, _, _ = get_info(goal)
        if name[0] != "~":
            asp_code += get_predicate_code("state", name, args, T="T")
            asp_code += ", "
            asp_code += get_predicate_code("not state",
                                           name, args, T="T", neg=True)
        else:
            asp_code += get_predicate_code("state",
                                           name[1:], args, T="T", neg=True)
            asp_code += ", "
            asp_code += get_predicate_code("not state", name[1:], args, T="T")
        asp_code += ", "

    asp_code += "time(T), not not_goal(T), "

    asp_code = asp_code[:-2] + ".\n"
    asp_code += "not_goal(T):- time(T), not goal(T). \n"

    # Constraint on goal that it should be reached within t_max time
    asp_code += ":- not_goal(T), T>=t_max.\n"

    # For each available action, choose only one action for each timestep if
    # we have not yet reached the goal
    asp_code += "1 { action(A, T) : available(A, T) } 1:- time(T), T<t_max, not_goal(T).\n"
    asp_code += """#minimize { T: action(_, T) }.\n"""

    asp_code += """#show action/2."""

    asp_code += "\n"

    output = get_optimal_answer_sets(asp_code)
    if not output:
        return output

    # Sort actions in right order
    new_output = dict()
    for action in output:
        action = action.split("(", 1)[1][:-1]
        number = action.split(",")[-1]
        action = action.split(")")[0] + ")"
        new_output[number] = action
    new_output = [new_output[k] for k in sorted(new_output)]

    indices = []
    # Store action as keys and args as value
    action_args = dict()

    # Transform back to action formatting from original
    for one_action in new_output:
        actions = planning_problem.actions
        # Obtain actions list for current planning problem
        actions = str(actions).replace("[", "").replace("]", "").split(", ")
        # Obtain actions args list for each action
        action_args = [a.args for a in planning_problem.actions]
        # Convert actions list to lowercase to compare
        actions_lower = [a.split("(", 1)[0].lower() for a in actions]
        if "(" in one_action:
            # Obtain action index for action in our output when it has args
            index = actions_lower.index(one_action.split("(", 1)[0])
        else:
            # Obtain action index for output if no args
            index = actions_lower.index(one_action.split(",", 1)[0])
        indices.append(index)

    # Transform back to action arg formatting from original using case dict
    output = []
    for (i, a) in zip(indices, new_output):
        # Obtain correct formatting for action and remove the old arguments
        new = actions[i].split("(", 1)[0]
        # If action has arguments, look up in case dict the correct formatting
        if "(" in a:
            new += "("
            for arg in a.split("(", 1)[1].split(")", 1)[0].split(","):
                new += case_dict[arg] + ","
            new = new[:-1] + ")"
        output.append(new)

    return output


def get_predicate_code(predicate, name, args, T=None, neg=False):
    """ Formats one part of a line of the program given name and arg, higher 
        order predicate and the time part of the higher order predicate as
        string, if applicable. """
    predicate_code = ""
    if predicate == "":
        # There is no higher order predicate and we use name as function.
        if args:
            predicate_code += "{}(".format(name)
            for arg in args:
                predicate_code += """{}, """.format(arg)
            predicate_code = predicate_code[:-2] + ")"
        else:
            predicate_code += "{}".format(name)
    else:
        if neg:
            # Use negation around name as higher order predicate.
            if args:
                predicate_code += "{}(neg({}(".format(predicate, name)
                for arg in args:
                    predicate_code += """{}, """.format(arg)
                if T:
                    predicate_code = predicate_code[:-2] + ")), {})".format(T)
                else:
                    predicate_code = predicate_code[:-2] + ")))"
            else:
                if T:
                    predicate_code += "{}(neg({}), {})".format(predicate, name, T)
                else:
                    predicate_code += "{}(neg({}))".format(predicate, name)
        else:
            # Normal case: use given predicate as predicate for name and format
            # args.
            if args:
                predicate_code += "{}({}(".format(predicate, name)
                for arg in args:
                    predicate_code += """{}, """.format(arg)
                if T:
                    predicate_code = predicate_code[:-2] + "), {})".format(T)
                else:
                    predicate_code = predicate_code[:-2] + "))"
            else:
                if T:
                    predicate_code += "{}({}, {})".format(predicate, name, T)
                else:
                    predicate_code += "{}({})".format(predicate, name)
    return predicate_code


def get_info(instance, convert_case=True):
    """ Return name, arguments, boolean list with where variables occur in args
        and return booleans that indicate amount of variables. """
    name = get_name(instance)
    args = convert_args(instance, convert_case)
    variables = [list(a)[0].isupper() for a in args]
    any_variables = any(a == True for a in variables)
    all_variables = all(a == True for a in variables)
    return name, args, variables, any_variables, all_variables


def get_name(instance):
    return str(instance).split("(", 1)[0].lower()


def get_action_code(action):
    """ Return formatted string with given action name and args. """
    action_code = ""
    args = convert_args(action.args)
    possible_code = ""
    if args:
        action_code += """{}(""".format(action.name.lower())
        prev_arg = []
        for arg in args:
            action_code += """{}, """.format(arg)
            if list(arg)[0].isupper():
                if prev_arg:
                    for p in prev_arg:
                        # Code for indicating that variables should be different
                        possible_code += "{} != {}, ".format(p, arg)
                prev_arg.append(arg)
        action_code = action_code[:-2] + ")"
    else:
        action_code += """{}""".format(action.name.lower())
    return action_code, possible_code


def convert_args(args, convert_case=True):
    """ Convert args expr to list of args. If convert_case True, convert to 
        formatting used in clingo. Variables start with uppercase letters in 
        clingo and objects start with lowercase."""
    if str(args) == "()" or "(" not in str(args):
        return []
    args = str(args).split("(", 1)[1][:-1].split(",")
    args = [a.strip() for a in args if a]
    new_args = []
    for arg in args:
        new_arg = list(arg)
        # Change variable when variable is equal to T variable
        if new_arg == ['t']:
            new_arg = ['tvar']
        if convert_case:
            if new_arg[0].isupper():
                for i, n_a in enumerate(new_arg):
                    new_arg[i] = new_arg[i].lower()
            elif new_arg[0].islower():
                new_arg[0] = new_arg[0].upper()
        new_arg = "".join(new_arg)
        new_args.append(new_arg)
    args = new_args
    return args


def get_optimal_answer_sets(program):
    """ Call clingo and get answer sets with optimality """
    control = clingo.Control()
    control.add("base", [], program)
    control.ground([("base", [])])

    # From ASP notebook example, get optimal models
    def on_model(model):
        global output
        if model.optimality_proven == True:
            sorted_model = [str(atom) for atom in model.symbols(shown=True)]
            sorted_model.sort()
            output = sorted_model

    control.configuration.solve.opt_mode = "optN"

    # Upper bound of 1 to get one optimal model
    control.configuration.solve.models = 1

    answer = control.solve(on_model=on_model)

    if answer.satisfiable:
        return output
    else:
        return None
    return output
