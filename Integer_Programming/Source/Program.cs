// See https://aka.ms/new-console-template for more information

/*
    Should i make stages of the unpacking of a struct?
    Raw to processed.
    When raw, storage efficient but retrieval is more costly.
    When processed, retrieval efficient but storage less efficient.
    More processed versions of the solution can point to less processed versions. Depending on how the
    data needs to be handled we use different levels of processing. Instantiating any of these types involves
    processing the data before use.
    They all implement the same abstract class so they are easily interchanged.

    Constraints->Tableau->Objective->Optimal Linear Tableau
                                    ->Pivot Matrix (Less processed than tableau therefore must multiply orginal constraint matrix by pivot matrix to get answer).
                                        We can take advantage of moments where the calculation has to be made by caching the tableau incase the properties of the
                                        tableau can be used for more purposes.
                                    ->Integer constraints->Mixed Integer Solution->Branch and Bound
                                                                                 ->Gomory Cut Method
    
    Remember that solving an integer program requires relaxing integer constraints and solving linear problems by
    adding linear equation constraints that approximate integer constraints.
        Only points on boundary of the decision space are ever solutions to problems with convex decision spaces and convex or concave objective functions.
        Therefore forcing these edges to have integer values for their decision variables is constraining the possible solutions to have integer values.     
    Eventually a linear program is found that has the same solution as the integer program.

    We need to add slack variables to make tableau
    We need to add objective function and goal or Objective
    We need to solve problem with object

*/

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.IO;
using System;
using System.Collections;
using System.Collections.Generic;
using Enumerable = System.Linq.Enumerable;
using MathNet.Numerics.Providers.LinearAlgebra;
using System.Net;
using System.Globalization;
using System.ComponentModel.DataAnnotations;

namespace Simplex{
    public class MainClass{
        /*
        public class Objective{
            Vector<double> 
            public Objective(){

            }
        }
        */

        /*
        public class BranchAndBoundTree{
            private SubproblemNode? root;
            private SubproblemNode? jumpRoot;
            private SubproblemNode? currentSubproblem;
            private PriorityQueue<SubproblemNode,double> jumpQueue;

        }    
        */
        public class SubproblemNode{
            public SubproblemNode? leftChild;
            public SubproblemNode? rightChild;
            public SubproblemNode? parent;
            public double upperBound;
            public bool fathomed {get; private set;}//True for leaf nodes where there are no more subproblems to be made. True for inner nodes  where both children are fathomed. 

            /* Proposition 1a: If a node B is a descendant of node A, then node B is a descendant of one of node A's children.
               Lemma 1b: If node B is a descendant of node A, then the path defined by (C1,...,Cm), where Cm is the root of the tree and Cn+1 is the parent of Cn
               must contain node B. B (- {Ci : i (- [1<i<=m]} 
            */                             
            /* Therorem 1: If an inner node is fathomed then all decendants of that node are fathomed.

                Assume there was an unfathomed leaf under a fathomed node, then its parent would be unfathomed because its parent 
                would have at least one unfathomed child. The parent's parent would be unfathomed for the same reason.
                There would therefore be a path of unfathomed nodes ending at the root of the tree.
                Since grandparent node we just assumed is fathomed must be on this path of unfathomed nodes, 
                there is a contradiction and the assumption that an unfathomed node can be under a fathomed node is false.
            */
            public  bool split {get; private set;}
            public Tableau problem;
            public SubproblemNode(SubproblemNode p,Tableau t,Vector<double> obj){
                problem = t;
                t.maximize(obj);
                upperBound = t.solution[t.dimension];
                parent = p;
                fathomed = false;
                split = false;
            }
            /*
                The "BackTrack" routine is meant to evoke the backtracking strategy for choosing subproblems to solve.
                Total reliance on backtracking is referred to as the Last In First Out (LIFO) strategy. It is the most rudimentary method for selecting the next subproblem to split and
                typically uses the most amount of computation time while being relatively efficient with memory usage, given that the parent nodes in the branch and bound tree are pruned 
                after descendant nodes are "fathomed". 

                Backtracking is often opposed to "Jumping" where the next chosen subproblem node is decided by the prooperties of the unfathomed subproblems. Often the subproblem with the 
                highest upperbound is chosen to be next when jumping. The integer solutions that come from these subproblems are a good bet for finding a high integer solution thus allowing
                for more subproblems to be implicitly processed. However, if one finds themeselves jumping all of the time the subproblem tree is more likely to be closer to balanced rather than
                degenerate. This means that the structure for ordering the unfathomed subproblems will be more full as thre will be far more leaves than inner nodes and the tree will likely have
                more nodes added to it between each node fathomed. Subproblems at deeper depths are more likely to be fathomed instead of split in comparison to ones that are closer to the root.
                As one adds constraints to a decision space, fewer solutions and therefore fewer integer solutions remain available. Either an integer solution is found or the deeper subproblems
                become infeasible. Moreover, subproblems that are at deeper depths are prone to having lower-upper bounds. Because deeper subproblems have smaller decision spaces they will likely
                lose decisions that have high values in the objective function. This means that if one only chooses the next subproblem based on its upperbound, they will be jumping often as the
                most recent nodes are likely have lower upper-bounds than the last subproblem from which were just split. THis is why commercial programs implementing the brach and bound algorithm 
                almost always use an approach that incorporates both strategies. Perhaps the newest subproblem no longer has the highest upper-bound in the tree but it is much closer to producing
                an incumbent solution than the problem with the highest upper-bound so it is smart to keep drilling down and jump to that subprobem later; once we arrive at that problem again
                the incumbent solution will more likely be higher and implicitly fathom more of the subproblems of that one we just considered jumping to.             

                Backtrack finds the first unfathomed node in "preorder". As in, if all of the nodes were listed in preorder the next node that backtrack would find
                would be the first unfathomed node to the left if the nodes were written in preorder. This algorithm works to find the most recently added subproblem
                in all cases if it is the only alogrithm used to choose the next subproblem node. Otherwise, it only works until the node that was last jumped to node (i will call it the "jump-root") 
                has "fathomed" set to true. If the previously "jumped from" node is not known then the "last node in", as far as the program knows, can be any leaf node on tree not under the jump-root. 
                
                With what I know about branch and bound trees I would have to store the previously jumped from nodes in a stack adding another parallel structure to the program with no concept of 
                how it will benefit time efficiency. My program will therefore only remember the node last jumped to. If that node is fathomed then it will likely be best to jump as the nodes 
                surrounding that subtree starting at the "jumproot" were abandoned for a good reason. If jumping leads back to said previosuly abandoned nodes they were chosen by a more sophisticated 
                hueristic than simply adhering to LIFO dogma at all times.     
            */
            public SubproblemNode? BackTrack(){  
                SubproblemNode? current = this;
                //First, if the current node is fathomed,(implying that there is no more work to be done on its children if they exist) jump to the parent
                //current shall never be in a fathomed node in the rest of the algorithm.
                if(this.fathomed){
                    current = this.parent;
                }
                if(current == null){
                    return null;
                }
                else{
                    do{//This loop makes the program climb towards the root of the tree until it can find a node that is the parent of an unfathomed leaf node.
                    if(current.leftChild.fathomed){
                        if(current.rightChild.fathomed){
                            /*
                                Every leaf node under this node is fathomed. Therefore, there is no use checking anything under this node.
                                For inner nodes, "fathomed" indicates that all subproblems under this subproblem are fathomed.
                            */
                            current.fathomed = true;
                            if(current.parent == null){
                                return null;
                            }
                            current = current.parent;
                        }
                        else{
                            current = current.rightChild;
                            break;
                        }
                    }
                    else{
                        current = current.leftChild;
                        break;
                    }
                }while(true);
                }
                
                //Once an unfathomed node is found climb the tree to the next subproblem leaf
                do{
                    if(current.leftChild == null){//Any node in the branch and bound algorithm that has children has two children; there are no nodes with one child.
                        return current;
                    }
                    if(!current.leftChild.fathomed){//Left is the first node to be checked to ensure the algorithm finding the next node in preorder
                        current = current.leftChild;
                    }
                    else{//Since the only way to enter this loop is when current is unfathomed,it is impossible to have both left and right be unfathomed.
                    //Also 
                        current = current.rightChild;
                    }
                }while(true);
            }
            public void Split(SubproblemNode l, SubproblemNode r){
                leftChild = l;
                rightChild = r;
                this.split = true;
            }
            public void Fathom(){
                this.fathomed = true;
                this.split = true;
            }


        }
        //This pivot function is common for any applications of matrices so it will remain outside of Tableau class.
        public static Matrix<double> pivot(Matrix<double> mat,int row,int col){
            double d = mat[row,col];
            Matrix<double> opMat = Matrix.Build.Dense(mat.RowCount,mat.RowCount,0);
            for(int i = 0;i<mat.ColumnCount;i++){
                mat[row,i] /= d; 
            }
            for(int i = 0;i<mat.RowCount;i++){
                if(i != row){
                    opMat[i,row] = -mat[i,col];
                }
                opMat[i,i] = 1;
            }
            opMat.Multiply(mat,mat);
            return mat;
        } 
        
        public struct Constraint{
            public enum Sign {eq,ge,le};
            public int dimension;
            public Sign sign;
            public List<double> lhs;
            public double rhs;
            public Constraint(){
                this.lhs = new();
            }
            public Constraint(List<double> l, double r,Sign s){
                this.lhs = new List<double>(l);
                this.rhs = r;
                this.sign = s;
                this.dimension = l.Count;
            }
            public Constraint(String line){
                this.lhs = new();
                dimension = 0;
                String buffer = "";
                double newItem;
                int j = 0;
                int i = 0;
                char cur = line[i];
                while(cur != '='){
                    if(cur == ' '){
                        if(Double.TryParse(buffer.Substring(0,j),out newItem)){
                            lhs.Add(newItem);
                            dimension++;
                            j = 0;
                            buffer = "";
                            i++;
                        }
                        else{
                            //throw Constraint coefficient is not numerical
                        }
                        
                    }
                    else{
                        buffer = buffer.Insert(j,line.Substring(i,1));
                        j++;
                        i++;
                    }
                    if(i == line.Length){
                        //throw no sign in constraint
                    }
                    cur = line[i];
                }
                i++;
                if(line[i] == ' '){
                    this.sign = Sign.eq;
                }
                else if(line[i] == '<'){
                    this.sign = Sign.le;
                    i++;
                }
                else if(line[i] == '>'){
                    this.sign = Sign.ge;
                    i++;
                }
                else{
                    //Throw Illegal sign in constraint.
                }
                i++;

                while(i<line.Length){
                    buffer = buffer.Insert(j,line.Substring(i,1));
                    i++;
                    j++;
                }
                if(Double.TryParse(buffer,out newItem)){
                    this.rhs = newItem;
                }
                else{
                    //throw Right hand side does not have numerical value.
                }
            }
            //We need to keep track of which variables are artificial and which ones are slack variables.
        }
        public enum Goal {max,min,none}; 
        public class Tableau{
            /*
                What if tableau was a seprate class from the initial constraints and basically everything that is common between programs which share constraints?
                We could simply keep the NXN matrix that performs the pivot operation from the constraints.
                We could make a programs objective class which has all of these prooperties associated with the objective that points back to the orginal constraints class
                which contains all structures that the objective class should have in common.
                We could make an abstract class for all of the ways one can use the original constraints. Single objective linear programming,
                multi-objective programming, parametric programming, mixed integer programming (where whichever decision variables are chosen to integral are listed in
                another structure.) etc. Perhaps the mixed integer programming class can be another abstract class where different children choose to make different decision
                variables integral.  
                Actually composition, instead of inheritance, would be much more sensical for this.
                Or maybe still make an abstract class but have the linear constraints be only a structure in the composition. The composition of the constraints is inherited.
                All of these implementations of the objective abstract class should at least share a common format for presenting their answers.

            */
            public List<Constraint> constraintList;
            private Matrix<double> constraints;
            private Matrix<double> initialConstraints;
            private List<int> basicColumns = new();//Each element in this list corresponds to its row in constraints; therefore the indexes on which to replace outgoing varaibles are easily determined: O(1) to replace and O(n) to iterate through
            //i.e. basicColumns[i] is row i
            private BitArray isNonBasic;//Saves time to have random access to whether or not a column is nonbasic.
            private SortedSet<int> nonbasicColumns = new();
            //What is the difference between the set of basic and nonbasic column numbers.
            private BitArray artificialColumns;
            private int artificialColumnCount;
            private int columnCount;
            private int rowCount;
            public int dimension;
            private bool feasible = false;
            
            /*  
                These two values below are used to explain the state of the tableau in that they give an objective
                and goal to explain why the constraints were pivoted in the way that they were. Otherwise, information
                is lost about what had happened before. 
            */
            private Vector<double>? objective;//
            public Goal goal;// 
            public Vector<double>? solution;
            public double value;
            public Tableau(List<Constraint> cons){
                constraintList = new List<Constraint>(cons);
                int negslack = 0;
                int slack = 0;
                int artificial = 0;
                this.dimension = cons[0].dimension;
                for(int i=0;i<cons.Count;i++){
                    if(cons[i].sign == Constraint.Sign.eq){
                        artificial++;
                    }
                    else if(cons[i].sign == Constraint.Sign.ge && cons[i].rhs > 0 || cons[i].sign == Constraint.Sign.le && cons[i].rhs<0){
                        artificial++;
                        negslack++;
                    }
                    else{
                        slack++;
                    }
                }
                
                int iCount = 0;
                int nCount = 0;
                this.artificialColumnCount = artificial;
                this.rowCount = cons.Count;
                this.columnCount = this.dimension+slack+negslack+artificial+1;//The last column of constraints is the rhs for the constraint.
                this.constraints = CreateMatrix.Dense<double>(this.rowCount,this.columnCount);
                this.initialConstraints = CreateMatrix.Dense<double>(this.rowCount,this.columnCount);
                artificialColumns = new BitArray(this.columnCount);
                isNonBasic = new BitArray(this.columnCount);
                int rhsSign;
                for(int i=0;i<cons.Count;i++){
                    rhsSign = (cons[i].rhs<0)?-1:1;
                    for(int j = 0;j<this.dimension;j++){
                        constraints[i,j] = rhsSign*cons[i].lhs[j];
                    }
                    constraints[i,this.columnCount-1] = rhsSign*cons[i].rhs;
                    if(cons[i].sign == Constraint.Sign.eq){
                        constraints[i,this.dimension+negslack+iCount] = 1;
                        artificialColumns[this.dimension+negslack+iCount] = true;
                        iCount++;
                    }
                    else if(cons[i].sign == Constraint.Sign.ge && cons[i].rhs > 0 || cons[i].sign == Constraint.Sign.le && cons[i].rhs<0){
                        constraints[i,this.dimension+nCount] = -1;
                        constraints[i,this.dimension+negslack+iCount] = 1;
                        artificialColumns[this.dimension+negslack+iCount] = true;
                        iCount++;
                        nCount++;
                    }
                    else{
                        constraints[i,this.dimension+negslack+iCount] = 1;
                        iCount++;
                    }
                }
                this.goal = Goal.none;
                this.constraints.CopyTo(this.initialConstraints);   
                for(int i = 0; i < this.dimension+negslack;i++){
                    this.isNonBasic[i] = true;
                    this.nonbasicColumns.Add(i);
                }
                for(int i = this.dimension+negslack; i < this.columnCount-1;i++){
                    this.basicColumns.Add(i);
                }                               
            }
            public void reset(){
                this.initialConstraints.CopyTo(constraints);
                int i;
                for(i = this.dimension;i<this.columnCount;i++){
                    if(this.constraints[0,i] == 1){//Find the column of the first variable in the identity section of the matrix.
                        break;
                    }
                }
                int j;
                for(j = 0;j<this.rowCount-1;j++){
                    basicColumns[j] = j+i;
                }
                isNonBasic.SetAll(false);
                for(j=0;j<i;j++){
                    isNonBasic[j] = true;
                    nonbasicColumns.Add(j);
                }
                return;
            }
            //Consider throwing a conditional exception for the simplex method. (Unbounded optimazation)
            public bool pivot(int col){//Returns whether or not the pivot is possible. 
                if(basicColumns.Find(x=> x == col) == -1){
                    Console.Write("Column ${col} is basic. Pivoting on a basic column is useless.");
                    return false;
                }
                if(col >= constraints.ColumnCount){
                    Console.Write("Column ${col} is out of bounds of the Tableau");
                    return false;
                }
                int row = -1;
                int outgoing = -1;
                double finalChange = double.MaxValue;
                double currentChange;
                for(int i = 0; i< this.constraints.RowCount; i++){
                    if(constraints[i,col] > 0){
                        currentChange = constraints[i,constraints.ColumnCount-1]/constraints[i,col];//If constraints[i,col] = 0, sets currentChange to infinity.
                        if(currentChange >= 0 && currentChange < finalChange){//Blands rule specifies that if there is a tie between the b/a ratios for tw0 or more rows then choose the row with the lowest index.
                            finalChange = currentChange;
                            row = i;
                        }
                    }
                    else if(constraints[i,constraints.ColumnCount-1] == 0 && constraints[i,col] != 0){ //If b=0 then multiplying both sides by -1 results in the same right hand side. Therefore coefficients that are negative can be treated as positive if b = 0.
                        finalChange = 0;
                        row = i;
                        break;//Its not going any lower than zero, just break. 
                    }
                }
                /*  
                    Degenerate points are points in the decision space where the number of hyperplanes of constraints that intersect at said point is greater
                    than the dimensionality of the decision space. The way one tells when the tableau is traversing upon a degenerate point is when one of the
                    basic variables is equal to zero. If procedure for exitting a degenerate points is not considered, the simplex algorithm might pivot back 
                    will be stuck with a subset of the same varaibles as its tableaus basic varaibles forever. This problem is called "cycling" in linear
                    programming terminology.

                    To counter this, this simplex algorithm implements bland's rule which means that when a degenerate point is reached the basic variable whose 
                    row is closest to the top (as in the row with basic variable equaling zero that corresponds to the lowest first index in the constraints variable
                    for "this") is always chosen. It might seem like an aribitrary rule but it prevents the tableau from pivoting in a cycle of the same basic variables.
                    I am not sure why this is the case, the books I read said it works with a citation and the video I found proving bland's rule is and hour long and
                    I don't think I want to try and understand an hour long proof when I sometimes struggle understanding very short proofs. 

                    Here is the hour long video proving bland's rule: https://www.youtube.com/watch?v=eGqv_vGaHS4&t=2060s 
                */
                if(finalChange == double.MaxValue){//If there was not a single row that had a nonnegative value in col selected the solution is unbounded.
                    Console.Write("entering basic variable is unbounded");
                    return false;
                }
                constraints = MainClass.pivot(constraints,row,col);
                //After a row is pivoted upon, swap outgoing basic(now nonbasic) varaibale in "basicColumns" with entering nonbasic(now basic) variable from "nonBasicColumns"
                outgoing = basicColumns[row];
                basicColumns[row] = col;
                isNonBasic[col] = false;
                nonbasicColumns.Add(col);
                isNonBasic[outgoing] = true;
                nonbasicColumns.Remove(outgoing);
                return true;
            }
            public double getZValue(Vector<double> obj){//Return z value of objective function in the current state the tableau is.
                if(this.solution == null){
                    return 0;
                }
                double ret = 0;
                for(int i= 0;i<this.dimension;i++){
                    ret += obj[i]*solution[i];
                }
                return ret;
            }
            public Vector<double> currentDecision(Vector<double> obj){
                Vector<double> ret = CreateVector.Dense<double>(this.dimension+1,0);
                double zvalue = 0;
                double varvalue = 0;
                //The value of the objective funciton at the  tableaus current postion is also calculated
                for(int i=0;i<basicColumns.Count;i++){
                    if(basicColumns[i] < this.dimension){//If basic column is structural variable
                        varvalue = constraints[i,this.columnCount-1];
                        ret[basicColumns[i]] = varvalue;
                        zvalue += obj[basicColumns[i]]*varvalue;
                    }
                }
                ret[ret.Count-1] = zvalue;
                return ret;
            }
            /*
                This function returns a vector that has the cj - zj values for each row. This allows decision makers to see how much increasing an entering variable
                will change the value of the objective function. 
            */

            public Vector<double> objectiveRow(Vector<double> obj){
                Vector<double> z = Vector.Build.Dense(constraints.ColumnCount);
                Vector<double> objective = CreateVector.Dense<double>(this.columnCount);
                Vector<double> newobjective = CreateVector.Dense<double>(this.columnCount);
                for(int i=0;i<this.columnCount;i++){
                    if(i<obj.Count){
                        objective[i] = obj[i];
                    }
                    else{
                        objective[i] = 0;
                    }
                }
                double zi = 0;
                double coefficient;
                double cj;
                for(int i = 0;i<constraints.ColumnCount;i++){
                    zi = 0;
                    for(int j = 0;j<constraints.RowCount;j++){
                        cj = objective[basicColumns[j]];
                        coefficient = constraints[j,i];
                        zi += constraints[j,i] * objective[basicColumns[j]];
                    }
                    newobjective[i] = objective[i]-zi;
                }
                return newobjective;
            }
            /*
                splitVariable implements the spitting chosen for the branch and bound algorithm. This rule determines which decision varaible
                should be constrained for the child subproblems of the chosen parent subproblem.  
            */
            public int splitVariable(Vector<double> obj){
                double value;
                double upper;
                double lower;
                double downPenalty;
                double upPenalty;
                double minPenalty = double.MaxValue;
                double maxPenalty = 0;
                int splitVariable = -1;
                double a;
                Vector<double> reducedCost = this.objectiveRow(obj);
                /*
                    Find the maximum of minimum penalties for adjusting each decision variable to the next lowest or highest integer.
                    Eachup or down penalty is estimation of how much the optimized value of the new subproblems will decrease in relation to
                    the optimal value of the subproblem that is being split. 
                */
                for(int index = 0; index < this.rowCount; index++){//Iterate through basic varaibles                    
                    if(basicColumns[index] < this.dimension && !isInteger(constraints[index,this.columnCount-1])){//No slack artificial variables need to be turned to integers. 
                        
                        value = constraints[index,this.columnCount-1];
                        upper = Math.Ceiling(value) - value;//The amount that the decision varaibale must change to reach the next integer above it.
                        lower = 1 - upper;//the same for the integer below the value of the decision variable.
                        minPenalty = double.MaxValue;
                        /*
                            What is basically happening in these loops is we are taking the row we got from this iteration of the outer loop and looking at its
                            linear equation. We start with the row as it shows up in the  tableau. With the column corresponding to the basic varaible as 1 and
                            that being juxatposed with the 'b' value on the other side of the equation. Each ai corrsponded to the coefficient for each nonbasic
                            variable in the constraint.

                                1 + a1 + a2 + a3 + ... = b

                            As previously stated, if all nonbasic variables are assume to be zero then this equation implies that the one basic variable in this
                            constraint is equal to 'b'. However, we want to set the basic variable to the nearest integer from above or below while still remaining
                            within the constraints of the decision space. So we have to set one of the nonbasic varaibles to be greater than zero to compensate for
                            the forcing of the basic variable in either direction.

                            At the same time we also want to see what happens to the value of our decision by and we have to look at the reduced-cost objective
                            function for our optimal tableau. 

                                c1 + c2 + c3 + ... = z 

                            Each of these coefficients is an approximation of how much the object function will change per incremment of 1 on a given nonbasic variable. 
                            In the simplex algorithm, the entering nonbasic variable for the next pivot is chosen based on which nonbasic variable is most likely to bring 
                            z closest to its optimal value by increasing its value from zero on each pivot. For the purposes of splitting, we will use this to find the 
                            "penalty cost" of adjusting a given basic variable to the nearest integers by compensating with each nonbasic variable. The formula for this
                            penalty cost on a given nonbasic variable is as follows:

                                p = ci*(f/ai)
                            
                            if f is the amount by which the basic variable has to change to reach either of the nearest integers and ai is the coefficient of the 
                            nonbasic variable in the constraint, then f/ai is the amount the nonbasic variable will have to change to compensate for the change in 
                            the basic variable to either of its nearest integers. ci is the coefficient for the reduced cost objective function for the optimal pivot
                            so the formula is clearly the amount the objective function will change when the basic variable is adjusted.

                            A distinction needs to be made between the up penalties and down penalties. An up penality is the cost to the objective function when 
                            a basic variable is adjusted to the nearest integer above it and a down enalty is the same but for the integer below. No nonbasic variable
                            has both an up and down penalty as the coefficent corresponding to it in the constraint is either negative or positive. Using a nonbasic
                            variable to compensate for a change in the basic variable must never entail decreasing its value as nonbasic variables are always zero
                            after a simplex pivot and all linear programs in canonical form must have a nonegative constraint on every variable whether basic, nonbasic,
                            decision, slack or artificial. This means that if ai is negative then the nonbasic variable can only be used to decrease the basic to
                            the nearest integer from below. Otherwise, it must increase the basic to the integer above. This implies that all penalties are postive
                            as f is negative when adjusting down and f is positive when adjusting up.

                            Once all penalties are calculated for adjusting a basic varaible using each nonbasic, the minimum of those penalties is found to find the
                            most optimistic outcome for the value of z when forcing the basic variable to be either of its nearest integer values. Once all of these
                            minimum values are calculated for each basic variable, the basic variable with the maximum of these minimums is chosen to be split upon.
                            The logic here is if we can ensure that both of the new subproblems have upper bounds that are as low as possible, then they are easier
                            to implicitly fathom.
                            
                             When we split on variable we are basically chosing a strip of the decision space to discard that will not contain 
                            any integer solutions. If we can find a strip that has as the highest values for z we can get an upper bound for our subproblems that
                            is closer to

                        */
                        foreach(int nonbas in nonbasicColumns){//Iterate through nonbasic variables.
                            a = constraints[index,nonbas];
                            if(a != 0){//If a is zero then there is no way we can compensate with the variable with that coefficient.
                                if(a > 0){
                                    downPenalty = reducedCost[nonbas]*(-lower/a);
                                    if(downPenalty <= minPenalty){
                                        minPenalty = downPenalty;
                                    }
                                }
                                else{
                                    upPenalty = reducedCost[nonbas]*(upper/a);
                                    if(upPenalty <= minPenalty){
                                        minPenalty = upPenalty;
                                    }
                                }     
                            }
                        }
                        if(minPenalty >= maxPenalty){
                            maxPenalty = minPenalty;
                            splitVariable = basicColumns[index];
                        }
                    }
                }
                return splitVariable;
            }

            //Make a list for basic columns refer to code complete for tips on how to fix the readabillity of code
            private bool _maximize(Vector<double> obj){                
                if(!this.Phase1()){//If the sum of artificial variables cannot be set to zero, then the problem is infeasible.
                    feasible = false;
                    return false;
                }
                solution = this.currentDecision(obj);
                bool bounded = true;
                Vector<double> w = this.objectiveRow(obj);
                //Vector<double> w0 = CreateVector.Dense<double>(this.columnCount);
                int firstPositive;
                double max = Double.MinValue;
                int maxIndex = -1;
                /*
                    If the program has made it past phase 1, then the only thing that can make pivot return false is if the optimal
                    solution is unbounded.
                */
                while(bounded){
                    max = Double.MinValue;
                    maxIndex = -1;
                    firstPositive = -1;
                    /*
                        Interesting thing to point out about pivoting on the column with the highest reduced-cost coefficient is that it is
                        commonly presented as the pivot that will guarantee the most change in the objective value. This is untrue. The 
                        reduced-cost only say how much the objective function will change per unit of increase in the varaible that corresponds
                        to that column. It says nothing about how much that variable can change. There is always the chance that other variables
                        to pivot on can change so much more than the variable corresponding to the highest reduced-cost that it causes the 
                        objective function to increase more. If we were to truely verify which pivot would increase the objective the most, it
                        would be extremely costly. The ammount of time saved by only pivoting on that variable everytime is far outweighed by the 
                        time spent on finding it.  
                    */
                    for(int i = 0;i<this.columnCount-1;i++){
                        if(isNonBasic[i] && !artificialColumns[i]){//If the column corresponding to i is not basic i.e. not in "basicColumns" then it is worth considering as an entering variable. 
                            if(w[i] > max){//Find the coefficient in the reduced-cost equation with the maximum value. Or that promises the greatest increase in the objective function.
                                max = w[i];
                                maxIndex = i;
                            }
                        }
                        if(firstPositive == -1 && max >= 0){//The column with the first positive value is crucial to know for following Bland's rule.
                            firstPositive = i;
                        }
                    }
                    if(max <= 0){//If the largest increase in z achieveable is negative or zero, the current point is optimal.
                        break;
                    }
                    if(this.constraints.Column(this.columnCount-1).Exists(x=>x==0)){//If there is a z
                        bounded = this.pivot(firstPositive);
                    }
                    else{
                        bounded = this.pivot(maxIndex);
                    }
                    w = this.objectiveRow(obj);
                }
                if(!bounded){
                    return false;//Do something to indicate undboundedness here.
                }
                return true;
            }

            public bool minimize(Vector<double> obj){
                if(objective != null){
                    if(this.objective.Equals(obj) && this.goal == Goal.min){
                        return feasible;
                    }
                    else{
                        this.reset();
                    }
                }
                bool ret = this._maximize(-1*obj);
                this.feasible = ret;
                //We must store return value temporarily because other properties of the tableau have to change independently of what can be done in private maximize. 
                this.objective = obj;
                this.goal = Goal.min;
                this.solution = this.currentDecision(obj);
                return ret;
            }
            public bool maximize(Vector<double> obj){
                if(objective != null){
                    if(this.objective.Equals(obj) && this.goal == Goal.max){
                        return feasible;
                    }
                    else{
                        this.reset();
                    }
                }
                bool ret = this._maximize(obj);
                this.feasible = ret;
                //We must store return value temporarily because other properties of the tableau have to change independently of what can be done in private maximize.
                this.objective = obj;
                this.goal = Goal.max;
                this.solution = this.currentDecision(obj);
                return ret;
            }
            /*
                If the linear program starts with artificial variables, a "phase1" needs to be performed to reduce the value of the artificial variables to
                zero. If the sum of the artificial variables cannot be brought to zero with setting any other variables to negative values then the system
                defining the linear equation is inconsistant and the feasible region is empty.
            */ 
            public bool Phase1(){
                if(this.artificialColumnCount == 0){
                    return true;
                }
                Vector<double> w = CreateVector.Dense<double>(this.columnCount);//The sum of artifical varaibles referred to as "w" in operations planning
                Vector<double> w0 = CreateVector.Dense<double>(this.columnCount);
                /*  This loop adds up each row with artifical variables to get the equation:

                        a1x1 + a2x2 + ... + anxn + xn+1 + ... + xm = b

                    Where n is the number of non-artificial variables and m-n is the number of artificial variables.
                    The sum of {xk : k (= [n+1,m]} is the sum of the artificial variables and will be refered to as "w". 
                    Each varaible in {ak : k (= [1,n]} is the sum of coefficients to variable k from each row that has an artificial variable. 
                    
                     
                */
                for(int i = 0;i<this.rowCount;i++){
                    if(artificialColumns[this.basicColumns[i]]){
                        w += constraints.Row(i);
                        w0 += constraints.Row(i);
                        w[basicColumns[i]] = 0;
                        w0[basicColumns[i]] = 0;
                    }
                }
                 
                /* 
                z squared (w[this.columnCount]) is negative so when operations are done on it to minimize it really means it should be maximized
                The equation that defines w is:

                        w = b - (a1x1 + ... anxn)
                    
                    Given that each variable in {xk : k(= [1,n]} is nonbasic and therefore  implying that w is equal to b in basic solutions,
                    this equation indicates that increasing xk where ak is positive will decrease the value of w.
                    Each pivot will increase the variable that will minimize b which will have the maximum value for {ak : k (= [1,n])}

                */
                while(w[this.columnCount-1] > 0){
                    double max = Double.MinValue;
                    double min = Double.MaxValue;
                    int maxIndex = -1;
                    int minIndex = -1;
                    int firstPositive = -1;
                    int chosen = -1;
                    for(int i = 0;i<this.columnCount-1;i++){
                        if(firstPositive == -1 && w[i] > 0){
                            firstPositive = i;
                        }
                        if(!basicColumns.Exists(x=>x==i)){
                            if(w[i] > max){//Find max change from nonbasic columns
                                max = w[i];
                                maxIndex = i;
                            }
                            if(w[i] < min){
                                min = w[i];
                                minIndex = i;
                            }
                        }
                    }
                    if(constraints.Column(this.columnCount-1).Exists(x=>x==0)){
                        /*Bland's Rule if you run into a degenerate solution to the system, pivot on the first column from the left with a positve objective row value
                        and the first row from the top.
                        */
                        chosen = firstPositive;
                    }
                    else{
                        chosen = maxIndex;
                    }
                    if(w[this.columnCount-1] != 0 && max <= 0){//There are no pivots that decrease the value of the sum of artifical variables and the sum has not reached zero.
                        //Throw no feasible solution exception
                        return false;
                    }
                    else{
                        this.pivot(chosen);
                        w = this.objectiveRow(w0);
                    }
                }
                return true;
            }
        public void printTableau(){
            Console.WriteLine(this.constraints);
        }
        public void printTableau(Vector<double> obj){
            Console.WriteLine(this.constraints);
            Console.WriteLine(objectiveRow(obj));
        }
    }
        /*
            WAIT A SECOND, HOW DO WE DEEP COPY A LIST SO WE DONT END UP SHARING CONSTRAINTS ACROSS ALL BRANCHES?
        */
        static Tableau newConstraints(Tableau tab,List<Constraint> con){
            List<Constraint> newCon = new List<Constraint>(tab.constraintList);
            foreach(Constraint c in con){
                newCon.Add(c);
            }
            return new Tableau(newCon);
        }
        static Tableau newConstraint(Tableau tab,Constraint con){
            List<Constraint> newCon = new List<Constraint>(tab.constraintList);//Shallow copies are sufficient. Constraints aren't meant to be mutable. If one wants to change constraints
            
            newCon.Add(con);                                                        //just replace the constraint completely. 
            return new Tableau(newCon);
        }

        //If the number is close enough to an integer value then the objective attained wont change so much when rounding.
        static bool isInteger(double input){
            return input - Math.Floor(input) < .00001 || Math.Ceiling(input) - input < .00001;
        }
    


        /*
            A linear integer program has the same types of objectives and contraints that an integer program has with the exception that the
            solutions to an integer program have to have integers. This often starts by solving the problem as if it was a linear program
            and then adding constraints to that problem based on the solution to the orgiginal problem. The addded constraints are intended 
            to force the optimal solutions of the linear program to have integer soltions while still being slovable using linear programming.

            IntegerSolve is an implmenetation of a branch and bound algorithm. First the problem is solved as if it were a linear program, 
            with the integer constraints "relaxed". Then the program then choses an arbitrary integer solution by rounding up or down decision 
            variables (xi: 1<=i<=m(dimension of decision space)) that output z until a feasible integer solution is found. This integer solution 
            is called the incumbent solution and is not intended to be a final solution but it will allow the program to implicitly fathom more 
            subproblems. 

            Once the first step of the algorithm is finished, a "splitting rule" is used to determine which of decision variables (xi) that are
            not integers should be split upon. This split decision variable will be 'xn'. In "splitting" the orginal linear problem, the program 
            simply makes two new problems. One where xn is constrained to be greater than or equal next integer above xn in the previous solution 
            and one where xn is constrained to be less than or equal. Then these subsolutions are solved and split in the same way.

            Since branch and bound is an optimization technique we can ignore subproblems that have a less optimal solution than the incumbent
            solution. If a subproblem has a lower optimzal solution to its linear program than the incumbent solution then will not be an integer 
            solution more optimal than the incumbent solution. The linear program finds the optimal solution possible in the entire decision
            space regardless of whether or not it is an integer solition. There will, therefore, not be any integer solution inside the decision
            space of the subproblem that is more optimal. This subproblem is not investigated further and is marked as fathomed". When subproblems
            are fathomed by having a relaxed solution that is less optimal than the incumbent solution they are considered "implicitly" fathomed. 

            If one of the subproblems of the integer program is found to have an integer solution to its linear program and that solution is
            more optimal than the incumbent solution, the incumbnent solution is replaced. Now more subproblems can be fathomed. Futhermore,
            that solution is the optimal soultion for that subproblem's decision space so that subproblem does not need to be split anymore 
            and is marked as fathomed.

            If a subproblem is found to be infeasible (in other words there are no points in the decision space that satisfy the constraints of
            the subproblem) that subproblem is marked as fathomed.

            The algorithm terminates once all of the subproblems are marked as fathomed.
        */

        /*
            The reason that integerMaximize returns the solution instead of simply changing some tableau object is because I was too lazy to make a similar
            object for intger problems. This makes working with both integer and linear programs in this software more complicated.  


            _integerMaximize has to construct new programs from the ground up as constraints are the most basic objects in the raw to
            processed hierarchy. Constraints must be given to a tableau object and it will return a completely new tableau with the new constraints added.

        */
        static void splitSubproblem(SubproblemNode currentSubproblem,Vector<double> solution,Vector<double> obj){
            int split = currentSubproblem.problem.splitVariable(obj);//choose the next decision varaible (xi) to turn to integer [line xxx]
            List<double> lhs = new(); 
            double upper = System.Math.Ceiling(solution[split]);
            double lower = System.Math.Floor(solution[split]);
            
            //Make the new constraint to added to the two subproblems
            for(int i = 0;i<currentSubproblem.problem.dimension;i++){
                if(i == split){
                    lhs.Add(1);
                }
                else{
                    lhs.Add(0);
                }
            }
            SubproblemNode upperProblem = new SubproblemNode(currentSubproblem,newConstraint(currentSubproblem.problem,new Constraint(lhs,upper,Constraint.Sign.ge)),obj);
            SubproblemNode lowerProblem = new SubproblemNode(currentSubproblem,newConstraint(currentSubproblem.problem,new Constraint(lhs,lower,Constraint.Sign.le)),obj);
            currentSubproblem.Split(upperProblem,lowerProblem);
        }
        static Vector<double>? _integerMaximize(Tableau tab,Vector<double> obj){
            Vector<double> incumbent = CreateVector.Dense<double>(tab.dimension+1,0);
            incumbent[tab.dimension] = Double.MinValue;
            double value = 0;
            if(!tab.maximize(obj)){//Maximize returns false if there is no solution or the solution is undbounded.
                Console.Write("Objective funciton is unbounded");//HOW DO YOU DISTINGUISH BETWEEN UNBOUNDED AND NO SOLUTION?
                return null;
            }
            SubproblemNode? currentSubproblem = new SubproblemNode(null,tab,obj);
            Vector<double> objectiveRow;
            Vector<double> solution;
            //Begin branch and bound algorithm
            do{
                if(!currentSubproblem.problem.maximize(obj)){//If the current subproblem is infeasible, fathom.
                    currentSubproblem.Fathom();
                }
                else{
                    objectiveRow = currentSubproblem.problem.objectiveRow(obj);
                    solution = currentSubproblem.problem.currentDecision(obj);
                    value = solution[solution.Count-1];//Dont want to dereference value from solution every time.
                    if(value <= incumbent[incumbent.Count-1]){
                        currentSubproblem.Fathom();
                    }
                    else if(!solution.Exists(x=>!isInteger(x))){//If there is not a decision variable in the solution that is not integer (i.e the solution is integer)
                        if(value > incumbent[incumbent.Count-1]){
                            incumbent = solution;
                        }
                        currentSubproblem.Fathom();
                    }
                    else{//Split and enqueue new subproblems
                        splitSubproblem(currentSubproblem,solution,obj);
                    }
                }
                currentSubproblem = currentSubproblem.BackTrack();//Returns null if the queue is empty.
            }while(currentSubproblem != null);//The queue is not empty.
            return incumbent;
        }
        static Vector<double>? integerMaximize(Tableau tab,Vector<double> obj){
            return _integerMaximize(tab,obj);
        }
        static Vector<double>? integerMinimize(Tableau tab,Vector<double> obj){
            Vector<double> ret = _integerMaximize(tab,-1*obj);
            ret[ret.Count-1] *= -1;
            return ret;
        }

        static Tableau loadConstraints(String filename){
            StreamReader stream = new(filename);
            List<Constraint> constraints = new();
            string? line;
            while(stream.Peek() >= 0){
                line = stream.ReadLine();
                if(line == null){
                    break;
                }
                else{
                    constraints.Add(new Constraint(line));
                }
            }
            return new Tableau (constraints);
        }
        static Vector<double>? loadObjective(string filename){
            StreamReader stream = new(filename);
            if(stream.Peek() >= 0){
                string line = stream.ReadLine();
                string[] coefficients = line.Split(" ");
                int len = coefficients.Length;
                if(coefficients[len-1].CompareTo("z") != 0 || coefficients[len-2].CompareTo("=") != 0){
                    Console.WriteLine("Sign and right-hand side symbol do not indicate that file is objective function file.");
                    return null;
                }
                Vector<double> result = CreateVector.Dense<double>(coefficients.Length-2);
                double coef;
                for(int i = 0;i<coefficients.Length-2;i++){
                    if(double.TryParse(coefficients[i],out coef)){
                        result[i] = coef;
                    }
                    else{
                        Console.WriteLine("Element in left-hand side of equation is not numeric.");
                        return null;
                    }
                }
                return result;
            }
            else{
                Console.WriteLine("Objective file \"{0}\" cannot be opened.",filename);
                return null;
            }            
        }
        /*
            The purpose of this console is to allow user to play objectives against staic constraints.
            Thus objectives are seen as a property of constraints or a perspective on the constraiants.
            The constraints are pivoted to make new systems of equations that have the same set of soltions
            and give a different perspective on the constraints.
            The objective function is also manipulated but it is also easy to simply take the values of decision
            variables and plug them in to the orginal objective function to get the value of the current solution.

            With the constraints we must make a matrix which is less obvious

        */
        static void Main(String[] args){
            
            /*
            try{
                StreamReader stream = new(input);
                while(stream.Peek() >= 0){
                    constraints.Add(new Constraint(stream.ReadLine()));
                }
            }
            catch(IOException e){
                Console.WriteLine(e.Message);
            }
            */
            Tableau? tab = null;
            Vector<double>? objective= null;
            Vector<double>? result = null;
            bool quit = false;
            string? command;
            string[] arg;
            int column;
            string constraintFile = "";
            string objectiveFile = "";
            bool resultSeen = true;
            bool tableauSeen = true;
            bool objectiveSeen = true;
            bool constraintLoaded = false;
            bool objectiveLoaded = false;
            
            while(quit == false){
                Console.Write("<Simplex");
                if(constraintLoaded){
                    Console.Write(" constraints: {0}",constraintFile);
                }
                if(objectiveLoaded){
                    Console.Write(" objective: {0}",objectiveFile);
                }
                Console.Write("> ");
                command = Console.ReadLine();
                arg = command.Split(" ");
                //Now read the the command.
                if(arg[0].CompareTo("quit") == 0){
                    quit = true;
                }
                else if(arg[0].CompareTo("load") == 0){
                    if(arg[1].CompareTo("constraint") == 0){
                        try{
                            tab = loadConstraints(arg[2]);
                            constraintFile = arg[2];
                            constraintLoaded = true;
                            tableauSeen = false;
                        }
                        catch(FileNotFoundException){
                            Console.WriteLine("File: %s could not be opened.",arg[2]);
                        }
                    }
                    if(arg[1].CompareTo("objective") == 0){
                        try{
                        objective = loadObjective(arg[2]);
                        objectiveFile = arg[2];
                        objectiveLoaded = true;
                        objectiveSeen = false;
                        }
                        catch(FileNotFoundException){
                            Console.WriteLine("File: %s could not be opened and loaded.",arg[1]);
                        }
                    }   
                }
                else if(tab != null && objective != null){//These conditions imply that tab and objective are not null.
                    if(arg[0].CompareTo("phase1") == 0){
                        tab.Phase1();
                        tableauSeen = false;
                        resultSeen = false;
                    }
                    else if(arg[0].CompareTo("pivot") == 0){ //pivot column _row_
                        if(arg.Length > 1){
                            if(Int32.TryParse(arg[1], out column)){
                                if(arg.Length > 2){
                                    //tab.pivot(column,row);
                                    tableauSeen = false;
                                }
                                else{
                                    tab.pivot(column);
                                    tableauSeen = false;
                                }
                            }
                            else{
                                Console.Write("there is no such column");
                            } 
                        }
                        else{
                            Console.Write("Invalid command");
                        }
                    }
                    else if(arg[0].CompareTo("max") == 0){
                        tab.maximize(objective);
                        result = tab.solution;
                        tableauSeen = false;
                        resultSeen = false;
                    }
                    else if(arg[0].CompareTo("min") == 0){
                        tab.minimize(objective);
                        result = tab.solution;
                        tableauSeen = false;
                        resultSeen = false;
                    }
                    else if(arg[0].CompareTo("int") == 0){
                        if(arg[1].CompareTo("max") == 0){
                            result = integerMaximize(tab,objective);
                            resultSeen = false;
                        }
                        else if(arg[1].CompareTo("min") == 0){
                            result = integerMinimize(tab,-1*objective);
                            resultSeen = false;
                        }
                        else{
                            Console.WriteLine("Not expecting {0} after int",arg[1]);
                        }
                    }
                    else{
                        Console.Write("Invalid command");
                    }                   
                }
                else{
                    Console.Write("Either constraint or objective file not loaded.");
                }
                if(!tableauSeen && tab != null){
                        Console.WriteLine("Constraints in canonical form:");
                        tab.printTableau();
                        tableauSeen = true;
                    }
                if(!objectiveSeen && objective != null){
                    Console.WriteLine("Objective function:");
                    Console.WriteLine(objective);
                    objectiveSeen = true;
                }
                if(result != null && !resultSeen){
                    Console.WriteLine("Decision:");
                    Console.WriteLine(result);
                    resultSeen = true;
                }
                else if(result == null && !resultSeen){
                    Console.WriteLine("No solution was found.");
                } 
            }
        }
    }
}

/*
    Most of the topics I discuss in these comments are addressed in this book:

    Harvey M. Salkin, Kamlesh Mathur (1989) "Foundations of Integer Programming"

    This book introduced me to a surface level understanding of the topics:

    Michael W Carter, Camille C Price (2001) "Operations Research: a Practical Introduction"

    This is the textbook I used for my discrete mathematics. It goes over some of the terminology I used to explain the Branch and Bound tree:

    Kenneth A Ross, Charles B Wright (2003) "Discrete Mathematics"

    If I can't explain something in a way that I am satified with or I dont understand why a procedure in this program works
    or is recommended, I will cite the page number where I learned about it.
*/