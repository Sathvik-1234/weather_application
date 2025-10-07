This project is a dynamic, single-page weather application designed to provide users with a comprehensive and easy-to-understand weather dashboard.

From a user's perspective, its core functionality includes:

A 3-Day Forecast: It displays key weather data for the current day and the next two days.

Flexible Search: Users can search for any city in the world or use a one-click geolocation button to get the weather for their current location.

Detailed Information: Beyond just the temperature, it shows humidity, wind speed, weather alerts, and accurate local sunrise and sunset times.

On the technical side, the application is built with standard HTML, CSS, and JavaScript. I leveraged several key libraries to enhance functionality:

jQuery was used for efficient DOM manipulation and for handling asynchronous AJAX calls to the OpenWeatherMap API.

Moment.js was integrated for reliable date and time formatting.

SunCalc.js was used to accurately calculate the sunrise and sunset times based on geographic coordinates.

A key feature of the application is its dynamic and responsive UI. For instance, the page's background is a gradient that programmatically changes based on the average temperature, providing a subtle visual cue—cooler blues for cold weather and warmer oranges for hot weather.

Finally, a more complex feature is the historical weather section. This required making five parallel, asynchronous API calls for the past five days and ensuring the UI only updated after all calls were successfully completed, which was a great exercise in managing asynchronous operations.

Overall, it’s a robust client-side application that effectively demonstrates API integration, asynchronous JavaScript, and the creation of a dynamic, user-friendly interface.
