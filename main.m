clc;
clear;

% Read the binary map
% Generate_Random_Map
map = imbinarize(imread('random_map.bmp'));

selectionprompt = "0: Tournament Selection, 1: Roulette Wheel, 2: Rank based";
crossoverprompt = "0: one point crossover, 1: unifrom crossover";
mutationprompt = "0: swap mutation, 1: flip mutation";
selection = input(selectionprompt);
crossover = input(crossoverprompt);
mutation = input(mutationprompt);

% Define start and end points
startpoint = [1 1];
endpoint = [500 500];

% Number of random points to generate
numRandomPoints = 7;
population_size = 100;
itter = 100;

% Generate initial population of paths
population = zeros(population_size, 2 * numRandomPoints); % Additional column for distance

for i = 1:population_size
    % Generate random points for each chromosome
    randomPointsX = randi([1, size(map, 2)], 1, numRandomPoints);
    randomPointsY = randi([1, size(map, 1)], 1, numRandomPoints);

    % Sort x-coordinates and rearrange y-coordinates accordingly this is
    % only done for the initial population so that you get points that
    % generally tend towards the end point
    [sortedX, sortIdx] = sort(randomPointsX);
    sortedY = randomPointsY(sortIdx);

    % Store both x and y coordinates in the population matrix
    population(i, 1:2:end) = sortedX;
    population(i, 2:2:end) = sortedY;
    
end

%% distance
weight = 700;

fittest = zeros(itter, 1); %initialize vector to store fitness score of fittest individual each generation for plotting

tic
% crossover probability is 60%
crossoverProb = 0.6;
% mutation probability is 0.3
mutationProb = 0.3;
topIndividualsCount = floor(population_size / 3); % Top one-third of the population

for generation = 1:itter
    % Append start and end points to the current population
    expandedPopulation = appendStartAndEndPoints(population, startpoint, endpoint, numRandomPoints);

    % Calculate distances and hits for the current population
    currentTotalDistances = calculateChromosomeDistances(expandedPopulation);
    currentTotalHitsVector = calculateHitsForEachChromosome(expandedPopulation, map);
    
    backpen = calcbackpenalty(population,numRandomPoints);

    % Calculate fitness for the current population
    currentFitnessVec = 1./(currentTotalDistances + (currentTotalHitsVector) .*weight);

    % Combine current population with fitness values
    currentPopulationWithFitness = [population, currentFitnessVec];

    %Sort the population with fitness values
    currentwithsort = sortrows(currentPopulationWithFitness,size(currentPopulationWithFitness,2),"descend");

    fittest(generation, 1) = currentwithsort(1, end); %save score of fittest in this generation k for plotting

    % Initialize a new population
    % newPopulation = zeros(population_size, 2 * numRandomPoints);
    % take the top 1/3 of the population elitism
    newPopulation = currentwithsort(1:topIndividualsCount,1:end-1);

    % currentwithsort = sortedByFitness(1:topIndividualsCount, :);
    
    
    
    % index where the elitism starts
    population_new_num = topIndividualsCount;
    % how many chromosomes are picked for tournament
    tournamentSize = 3;

    while population_new_num <= population_size
        % Selection
        if selection == 0
            choice1 = tournamentSelection(currentFitnessVec, tournamentSize);
            choice2 = tournamentSelection(currentFitnessVec, tournamentSize);
        
        elseif  selection == 1
            currentFitnessVecWeight = currentFitnessVec./sum(currentFitnessVec);
            choice1 = RouletteWheelSelection(currentFitnessVecWeight);
            choice2 = RouletteWheelSelection(currentFitnessVecWeight);
        elseif selection == 2
            choice1 = rankBasedSelection(currentFitnessVec);
            choice2 = rankBasedSelection(currentFitnessVec);
        end
        
    
        % Extract chromosomes
        parent1 = population(choice1, :);
        parent2 = population(choice2, :);
    
        % Crossover
        
        if crossover == 0
             if rand < crossoverProb
                [offspring1, offspring2] = OnePointCrossover(parent1, parent2); % or OnePointCrossover

            else
                offspring1 = parent1;
                offspring2 = parent2;
            end
        elseif crossover == 1
           if rand < crossoverProb
                [offspring1, offspring2] = UniformCrossover(parent1, parent2); % or OnePointCrossover
            else
                offspring1 = parent1;
                offspring2 = parent2;
           end
        end
       
    
        if mutation == 0
        % Mutation
            if rand < mutationProb
                offspring1 = SwapMutation(offspring1);
                offspring2 = SwapMutation(offspring2);
            end
        elseif mutation == 1
            if rand < mutationProb
                offspring1 = FlipMutation(offspring1);
                offspring2 = FlipMutation(offspring2);
            end
        end
    
        % Add first offspring to the new population
        newPopulation(population_new_num, :) = offspring1;
        %increment the 
        population_new_num = population_new_num + 1;
    
        % Check if there is room for the second offspring
        if population_new_num <= population_size
            newPopulation(population_new_num, :) = offspring2;
            population_new_num = population_new_num + 1;
        end
    end

    %when the loop for generatiosn is done assign the newpopulation as
    %population 
    population = newPopulation;

end





% Append start and end points to the current population
    expandedPopulation = appendStartAndEndPoints(population, startpoint, endpoint, numRandomPoints);

    backpen = calcbackpenalty(population,numRandomPoints);

    % Calculate distances and hits for the current population
    currentTotalDistances = calculateChromosomeDistances(expandedPopulation);
    currentTotalHitsVector = calculateHitsForEachChromosome(expandedPopulation, map);

    % Calculate fitness for the current population
    currentFitnessVec = 1./(currentTotalDistances + (currentTotalHitsVector) .* weight);

    % Combine current population with fitness values
    currentPopulationWithFitness = [population, currentFitnessVec];

    %Sort the population with fitness values
    currentwithsort = sortrows(currentPopulationWithFitness,size(currentPopulationWithFitness,2),"descend");


% Display the best path after the final iteration

figure;
imshow(map);
hold on;

% Extract x and y coordinates for the best path
bestChromosome = currentwithsort(1, 1:end-1); % Best chromosome is the first one after sorting
x_coords = [startpoint(1), bestChromosome(1:2:end), endpoint(1)];
y_coords = [startpoint(2), bestChromosome(2:2:end), endpoint(2)];

% Plot the best path
line(x_coords, y_coords, 'Color', 'r', 'LineWidth', 1);
title('Best Path After Final Generation');
hold off;
toc

% plot fitness score of fittest individual each generation
t =1:1:itter; 
figure('Name','Fittness evolution','Numbertitle','off');
plot(t, fittest);
xlabel('generations'), ylabel('fitness score');
title ('Fittest individual each generation'); 




%% new distance function
function totalDistances = calculateChromosomeDistances(population)
    population_size = size(population, 1);
    numPoints = size(population, 2) / 2;

    % Reshape population to separate x and y coordinates for each point
    reshapedPopulation = reshape(population', [2, numPoints, population_size]);

    % Extract x and y coordinates
    xCoords = squeeze(reshapedPopulation(1, :, :));
    yCoords = squeeze(reshapedPopulation(2, :, :));

    % Calculate differences between consecutive x and y coordinates
    xDiffs = diff(xCoords);
    yDiffs = diff(yCoords);

    % Compute distances for each pair of points
    distances = sqrt(xDiffs.^2 + yDiffs.^2);

    % Sum distances for each chromosome
    totalDistances = sum(distances, 1)';
end

%% append 1 1 and 500 500

function expandedPopulation = appendStartAndEndPoints(population, startpoint, endpoint, numRandomPoints)
    population_size = size(population, 1);
    
    % Create arrays of start and end points replicated for the entire population
    startPoints = repmat(startpoint, population_size, 1);
    endPoints = repmat(endpoint, population_size, 1);

    % append start points, original population, and end points
    expandedPopulation = [startPoints, population, endPoints];
end
%% one point crossover
function [offspring1, offspring2] = OnePointCrossover(parent1, parent2)
    % Ensure the chromosomes are row vectors
    parent1 = reshape(parent1, 1, []);
    parent2 = reshape(parent2, 1, []);

    % Randomly choose the crossover point
    crossoverPoint = randi([2, numel(parent1) - 1]);

    % Perform one-point crossover to create two offspring
    offspring1 = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
    offspring2 = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
end
%% flip mutation
function flippedChromosome = FlipMutation(chromosome)

    %randomly pick two points within the chromosome
    point1 = randi([1, length(chromosome)]);
    point2 = randi([1, length(chromosome)]);

    %continue to check for point2 and point 1 being equal since they cant
    %be equal
    while point2 == point1
        %if they are equal regen random point for point 2
        point2 = randi([1, length(chromosome)]); % so that point1 != point2
    end

    if point2 < point1
        % Swap values so that point1 < point2
        temp = point1;
        point1 = point2;
        point2 = temp;
    end

    % Flip the segment between point1 and point2
    tobeflipped = chromosome(point1:point2);
    chromosome(point1:point2) = fliplr(tobeflipped);

    flippedChromosome = chromosome;
end
%% roulettewheelselection
function choice = RouletteWheelSelection(weights)
   
%the selection probability of each individual.
  accumulation = cumsum(weights);
  %rand num between 0 and 1 used to select chrom based on their proportion
  p = rand();
  % Initialize the chosen index as -1.
  chosen_index = -1;
  %iterate through all weights
  for index = 1 : length(accumulation)
      %set chosenidx as index if p is less than accumulation 
    if (accumulation(index) > p)
      chosen_index = index;
      break;
    end
  end
  choice = chosen_index;
end

%% Tournment selection
function selectedIndex = tournamentSelection(fitnessScores, tournamentSize)
    populationSize = length(fitnessScores);
    
    % Randomly select individuals for the tournament
    tournamentIndices = randi(populationSize, tournamentSize, 1);
    
    % Extract the fitness scores of the selected individuals
    tournamentFitness = fitnessScores(tournamentIndices);
    
    % Find the index of the individual with the highest fitness in the tournament
    [~, bestInTournament] = max(tournamentFitness);
    
    % Get the index of the winner in the original population
    selectedIndex = tournamentIndices(bestInTournament);
end

%% rank based selection
function selectedIndex = rankBasedSelection(fitnessScores)
    populationSize = length(fitnessScores);
    
    % Rank individuals based on fitness (higher fitness -> lower rank)
    [~, rank] = sort(fitnessScores, 'descend');

    % Assign selection probabilities (linear ranking)
    % Highest rank gets populationSize probability, lowest gets 1
    probabilities = populationSize - rank + 1;
    probabilities = probabilities / sum(probabilities);

    % Cumulative sum of probabilities
    cumulativeProbabilities = cumsum(probabilities);

    % Select an individual
    randomNum = rand();
    selectedIndex = find(cumulativeProbabilities >= randomNum, 1, 'first');
end

%% unifrom crossover
function [offspring1, offspring2] = UniformCrossover(parent1, parent2)
    % Ensure the chromosomes are row vectors
    parent1 = reshape(parent1, 1, []);
    parent2 = reshape(parent2, 1, []);

    % Initialize offspring
    offspring1 = zeros(1, length(parent1));
    offspring2 = zeros(1, length(parent2));

    % Perform uniform crossover
    for i = 1:length(parent1)
        if rand < 0.5
            offspring1(i) = parent1(i);
            offspring2(i) = parent2(i);
        else
            offspring1(i) = parent2(i);
            offspring2(i) = parent1(i);
        end
    end
end


%% swap mutation

function swappedChromosome = SwapMutation(chromosome)
%randomly select two points on chromosome
    point1 = randi([1, (length(chromosome)/2)-1]);
    point2 = randi([1, (length(chromosome)/2)-1]);

    %make sure that the two poitns not the same
    while point2 == point1
        point2 = randi([1, (length(chromosome)/2)-1]); % so that point1 != point2
    end

    %make a copy of the original chromosome
    copyofchrom = chromosome;
    %swap the two points in the chromosome
    chromosome(point1:point1+1) = copyofchrom(point2:point2+1);
    chromosome(point2:point2+1) = copyofchrom(point1:point1+1);
    % Flip the segment between point1 and point2
    

    swappedChromosome = chromosome;
end
%% hits with start and end points
function totalHitsVector = calculateHitsForEachChromosome(population, map)
    population_size = size(population, 1);
    numPoints = size(population, 2) / 2;
    totalHitsVector = zeros(population_size, 1);

    for chromIdx = 1:population_size
        % hitcounter2 = 0;

        xValues = population(chromIdx, 1:2:end-1);
        yValues = population(chromIdx, 2:2:end);
        counts2 = improfile(map, xValues, yValues);
        hitcounter2 = sum(counts2==0);
        

        % Store the total hits for the current chromosome
        totalHitsVector(chromIdx) = hitcounter2;
    end
end

%% going backwards penalty
function backpenalties = calcbackpenalty(chromosomes, numrandpoints)
    population_size = size(chromosomes, 1);
    backpenalties = zeros(population_size, 1);
    
    %iterate over all the chromosomes in the matrix
    for i = 1:population_size
        %initialise penalty as 0 fro each chromosome
        penalty = 0;
        % Iterate only over x-coordinates
        for j = 1:2:(numrandpoints*2-2)
            %calculate the next and current x coords
            currentx = chromosomes(i,j);
            nextx = chromosomes(i,j+2);

            %if the current x coordinate is larger than the next one then
            %add a penalty 
            if currentx < nextx
                penalty = penalty + 10;
            end
        end
        % return the penalty vector
        backpenalties(i) = penalty;
    end
end
