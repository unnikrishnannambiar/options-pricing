# options-pricing

I am analysing options for the NIFTY 50 Index. Since index options in India follow the European style where we can only execute the option after the specified time, I will be using the famous Black-Scholes pricing formula since some of the model's key assumptions fit perfectly to our asset. European and Plain Vanilla.

I am planning to use an alternative to pandas to pickle and use options chains of Nifty 50 but since pandas has stopped support for options chain collection from Yahoo Finance, I am yet to find a solution.

However, I have taken the time to code in the Greeks to analyse various price movements with respect to fundamental parameters.
