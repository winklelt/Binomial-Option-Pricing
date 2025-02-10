// part(a) European Call option pricing using Binomial model
// part(b) American Call option pricing using Binomial model
// part(c) find and plot the optimal early exercise boundary for American Call option based on lowest stock price
// part(d) find the optimal exercise bounday for American Call option based on linear interpolation
// part(e) plot the optimal exercise boundary for American Call option from (d)
// part(f) improvement of the linear interpolation method from (d)

#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

// Function to price European and American options using a binomial tree
double binomialOption(bool isAmerican, int m, double S0, double K, double T, double sigma,
    double r, double y, vector<vector<double>>& exerciseBoundaryPartC,
    vector<vector<double>>& interpolatedBoundaryPartD,
    vector<vector<double>>& interpolatedBoundaryPartF)
{
    double dt = T / m; // timestep in Binomial model
    double u = exp(sigma * sqrt(dt)); // up factor
    double d = exp(-sigma * sqrt(dt)); // down factor
    double discount = exp(-r * dt); // discount factor (applied at each step as in your original code)
    double q_u = (exp((r - y) * dt) - d) / (u - d); // risk-neutral probability (up)
    double q_d = 1 - q_u; // risk-neutral probability (down)

    vector<vector<double>> values(m + 1); // matrix to store option values
    exerciseBoundaryPartC.resize(m);      // Part (c): store lowest stock price for early exercise			
    interpolatedBoundaryPartD.resize(m);    // Part (d): store linear-interpolated exercise boundary
	interpolatedBoundaryPartF.resize(m); // Part (f): store quadratic-interpolated exercise boundary


    // Initialize the values at maturity
    for (int i = 0; i <= m; ++i) {
        double stockPrice = S0 * pow(u, i) * pow(d, m - i); // stock price after i up and (m-i) down moves
        values[m].push_back(max(stockPrice - K, 0.0));        // payoff of call option
    }

    // Backward induction for option pricing
    for (int j = m - 1; j >= 0; --j)
    {
        values[j].resize(j + 1);              // allocate space for j+1 nodes at time step j
        vector<double> stockPrices(j + 1);      // store stock prices at time j
        vector<double> netValues(j + 1);        // store net (exercise - continuation) values at each node
        double minExercisePrice = INFINITY;     // For part (c): lowest stock price where early exercise is optimal

        // Compute payoff at each node and identify early exercise conditions
        for (int i = 0; i <= j; ++i) {
            double stockPrice = S0 * pow(u, i) * pow(d, j - i);
            stockPrices[i] = stockPrice;
            
            // Calculate continuation value
            double continuation = (q_u * values[j + 1][i + 1] + q_d * values[j + 1][i]) * discount;

            // Early exercise value
            double exercise = max(stockPrice - K, 0.0);
            netValues[i] = exercise - continuation; // net value = early exercise value minus continuation
			
           
            // American option: choose the maximum between exercising early and continuing
            if (isAmerican) {
                values[j][i] = max(exercise, continuation);
                if (exercise > continuation) {
                    if (stockPrice < minExercisePrice) {
                        minExercisePrice = stockPrice; // record lowest stock price triggering early exercise (part c)
                    }
                }
            }
            else {
                values[j][i] = continuation;
            }
        }

        // Part (c): Save the lowest stock price for early exercise at time j (or NAN if none)
        if (minExercisePrice < INFINITY) {
            exerciseBoundaryPartC[j] = { j * dt, minExercisePrice };
        }
        else {
            exerciseBoundaryPartC[j] = { j * dt, NAN };
        }

        // Part (d): Calculate the interpolated boundary S*(j·Δt).
        double S_star = NAN;

        // Find adjacent nodes with a sign change (negative → positive)
        for (int i = 1; i <= j; ++i) {
            if (netValues[i] >= 0 && netValues[i - 1] < 0) {
                double S_low = stockPrices[i - 1];
                double S_high = stockPrices[i];
                double net_low = netValues[i - 1];
                double net_high = netValues[i];

                // Linear interpolation
                double t = -net_low / (net_high - net_low);
                S_star = S_low + t * (S_high - S_low);
                break;
            }
        }

        interpolatedBoundaryPartD[j] = { j * dt, S_star };


        // Part (f): Quadratic interpolation for EEB
        S_star = NAN;
        for (int i = 1; i <= j; ++i) {
            if (netValues[i] >= 0 && netValues[i - 1] < 0) {
                // Check if we have at least three points for quadratic fit
                if (i >= 2) {
                    double x0 = stockPrices[i - 2];
                    double y0 = netValues[i - 2];
                    double x1 = stockPrices[i - 1];
                    double y1 = netValues[i - 1];
                    double x2 = stockPrices[i];
                    double y2 = netValues[i];

                    // Quadratic coefficients (ax² + bx + c = 0)
                    double denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
                    if (denom != 0) {
                        double a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
                        double b = (x2 * x2 * (y0 - y1) + x0 * x0 * (y1 - y2) + x1 * x1 * (y2 - y0)) / denom;
                        double c = (x1 * x2 * (x1 - x2) * y0 + x0 * x2 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom;

                        // Solve quadratic equation
                        double discriminant = b * b - 4 * a * c;
                        if (discriminant >= 0) {
                            double sqrt_d = sqrt(discriminant);
                            double root1 = (-b + sqrt_d) / (2 * a);
                            double root2 = (-b - sqrt_d) / (2 * a);

                            // Check which root lies in [x1, x2]
                            if (root1 >= x1 && root1 <= x2) S_star = root1;
                            else if (root2 >= x1 && root2 <= x2) S_star = root2;
                        }
                    }
                }

                // Fallback to linear interpolation if quadratic fails
                if (isnan(S_star)) {
                    double S_low = stockPrices[i - 1];
                    double S_high = stockPrices[i];
                    double net_low = netValues[i - 1];
                    double net_high = netValues[i];
                    double t = -net_low / (net_high - net_low);
                    S_star = S_low + t * (S_high - S_low);
                }
                break; // Process only the first sign change
            }
        }
        interpolatedBoundaryPartF[j] = { j * dt, S_star };
    }


    // Return the option price at the root node
    return values[0][0];
}

// PS2_Q3: Binomial coefficient function for European Call option pricing (closed-form binomial)
double binomialcoeff(int n, int k)
{
    if (k > n) return 0;

    // Use symmetry to reduce computation
    if (k > n - k) k = n - k;

    double c = 1;
    for (int i = 0; i < k; ++i) {
        c *= (n - i);
        c /= (i + 1);
    }
    return c;
}

// PS2_Q3: Pricing European Call option using the closed-form binomial formula
double binomialEuropeanCall(
    int m,         // number of timesteps
    double s0,     // initial stock price
    double k,      // strike price 
    double T,      // maturity in years
    double sigma,  // volatility
    double r,      // risk-free rate
    double y       // dividend yield rate
)
{
    double dt = T / static_cast<double>(m);
    double u = exp(sigma * sqrt(dt)); // up factor
    double d = exp(-sigma * sqrt(dt)); // down factor
    double discount = exp(-r * T);     // discount factor over the entire period
    double q_u = (exp((r - y) * dt) - d) / (u - d);
    double q_d = 1 - q_u;

    double value = 0;
    // Sum over all possible up-moves at maturity
    for (int i = 0; i <= m; ++i) {
        double st = s0 * pow(u, i) * pow(d, m - i);  // stock price after i up moves
        double payoff = max(st - k, 0.0);
        double prob = binomialcoeff(m, i) * pow(q_u, i) * pow(q_d, m - i);
        value += payoff * prob;
    }
    return discount * value;
}

int main()
{
    // Parameter initialization
    double S0 = 100.0;
    double k = 100.0;
    double T = 1.0;
    double sigma = 0.2;
    double r = 0.04;
    double y = 0.02;
    int m = 100; // number of time steps

    // Boundary storage for parts (c) and (d)
    vector<vector<double>> exerciseBoundaryPartC;      // Part (c): discrete early exercise boundary
    vector<vector<double>> interpolatedBoundaryPartD;    // Part (d): interpolated exercise boundary
    vector<vector<double>> interpolatedBoundaryPartF; // For quadratic interpolation
    
    // Part (a): European Call option pricing using the binomial tree (PS4_Q1 method)
    double ecPrice_PS4Q1 = binomialOption(false, m, S0, k, T, sigma, r, y,
        exerciseBoundaryPartC, interpolatedBoundaryPartD, interpolatedBoundaryPartF);
    cout << "European Call Price (PS4_Q1 Binomial Tree): " << ecPrice_PS4Q1 << endl;

    // Compare with European Call option pricing from PS2_Q3 (closed-form binomial formula)
    double ecPrice_PS2Q3 = binomialEuropeanCall(m, S0, k, T, sigma, r, y);
    cout << "European Call Price (PS2_Q3 Closed-form Binomial): " << ecPrice_PS2Q3 << endl;
    cout << "Difference: " << fabs(ecPrice_PS4Q1 - ecPrice_PS2Q3) << "\n\n";
    
    // Part (b): American Call option pricing using the binomial tree
    double acPrice = binomialOption(true, m, S0, k, T, sigma, r, y,
        exerciseBoundaryPartC, interpolatedBoundaryPartD, interpolatedBoundaryPartF);
    cout << "American Call Price: " << acPrice << endl;
    cout << "Early exercise premium (AC - EC): " << acPrice - ecPrice_PS4Q1 << "\n\n";

    // Part (c): Output the discrete early exercise boundary (lowest stock price for early exercise)
    ofstream file1("exercise_boundary_part_c.csv");
    file1 << "Time,Price\n";
    for (const auto& point : exerciseBoundaryPartC) {
        if (!isnan(point[1])) {
            file1 << point[0] << "," << point[1] << "\n";
        }
    }
    file1.close();

    
	
    // Part (d): Export the interpolated exercise boundary data (optimal boundary via linear interpolation)
    ofstream file2("exercise_boundary_part_d.csv");
    file2 << "Time,Price\n";
    for (const auto& point : interpolatedBoundaryPartD) {
        if (!isnan(point[1])) {
            file2 << point[0] << "," << point[1] << "\n";
        }
    }
    file2.close();

    // ========================
    // Part (e): Output all stock price nodes in the binomial tree
    // Create a CSV file "stock_prices.csv" with columns: Time, NodeIndex, StockPrice.
    // Note: The stock price at node (j, i) is S0 * u^i * d^(j-i)
    double dt = T / m;
    double u = exp(sigma * sqrt(dt));
    double d = exp(-sigma * sqrt(dt));
    ofstream file3("stock_prices.csv");
    file3 << "Time,NodeIndex,StockPrice\n";
    for (int j = 0; j <= m; ++j) {
        double timePoint = j * dt;
        for (int i = 0; i <= j; ++i) {
            double stockPrice = S0 * pow(u, i) * pow(d, j - i);
            file3 << timePoint << "," << i << "," << stockPrice << "\n";
        }
    }
    file3.close();


    // Export part (f) data
    ofstream file("exercise_boundary_part_f.csv");
    file << "Time,Price\n";
    for (const auto& point : interpolatedBoundaryPartF) {
        if (!isnan(point[1])) {
            file << point[0] << "," << point[1] << "\n";
        }
    }
    file.close();
    

    return 0;
}
