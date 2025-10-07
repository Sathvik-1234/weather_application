var API_KEY = "6841e5450643e5d4ff59981dbf58944e";

// -- On load --
$(document).ready(function() {
    if (!navigator.geolocation) {
        $('#geolocation').hide();
    }

    var city = document.location.hash ? document.location.hash.substr(1) : "London";
    var date = moment();

    for (var i = 0; i < 3; i++) {
        var day = $("#meteo-day-" + (i + 1));
        day.find(".name").text(date.format("dddd"));
        day.find(".date").text(date.format("DD/MM"));
        date = date.add(1, 'days');
    }

    var loading = $('#search-loading');
    loading.attr('class', 'loading inload');

    getMeteoByCity(city, function(data, error) {
        if (error == null) {
            displayMeteo(data);
        } else {
            var meteoTitle = $('#meteo-title span');
            meteoTitle.html('City <span class="text-muted">' + city + '</span> not found');
        }
        setTimeout(function() {
            loading.attr('class', 'loading');
        }, 500);
    });
});

// -- Core --
$("#meteo-form").submit(function(event) {
    event.preventDefault();
    var loading = $('#search-loading');
    loading.attr('class', 'loading inload');

    var city = event.currentTarget[0].value;
    getMeteoByCity(city, function(data, error) {
        if (error == null) {
            displayMeteo(data);
        } else {
            var meteoTitle = $('#meteo-title span');
            meteoTitle.html('City <span class="text-muted">' + city + '</span> not found');
        }
        setTimeout(function() {
            loading.attr('class', 'loading');
        }, 500);
    });
});

$("#geolocation").click(function(event) {
    navigator.geolocation.getCurrentPosition(function(position) {
        var loading = $('#search-loading');
        loading.attr('class', 'loading inload');

        var lat = position.coords.latitude;
        var lon = position.coords.longitude;

        getMeteoByCoordinates(lat, lon, function(data, error) {
            if (error == null) {
                displayMeteo(data);
            } else {
                var meteoTitle = $('#meteo-title span');
                meteoTitle.html('Can\'t get meteo for your position');
            }
            setTimeout(function() {
                loading.attr('class', 'loading');
            }, 500);
        });
    });
});

function getMeteoByCity(city, callback) {
    $.ajax({
        url: "https://api.openweathermap.org/data/2.5/forecast?q=" + city + "&APPID=" + API_KEY,
        success: function(data) {
            callback(data, null);
        },
        error: function(req, status, error) {
            callback(null, error);
        }
    });
}

function getMeteoByCoordinates(lat, lon, callback) {
    $.ajax({
        url: "https://api.openweathermap.org/data/2.5/forecast?lat=" + lat + "&lon=" + lon + "&APPID=" + API_KEY,
        success: function(data) {
            callback(data, null);
        },
        error: function(req, status, error) {
            callback(null, error);
        }
    });
}

function displaySunriseSunset(lat, long) {
    var date = moment();
    for (var i = 0; i < 3; i++) {
        var times = SunCalc.getTimes(date, lat, long);
        var sunrise = pad(times.sunrise.getHours(), 2) + ':' + pad(times.sunrise.getMinutes(), 2);
        var sunset = pad(times.sunset.getHours(), 2) + ':' + pad(times.sunset.getMinutes(), 2);

        var day = $("#meteo-day-" + (i + 1));
        day.find('.meteo-sunrise .meteo-block-data').text(sunrise);
        day.find('.meteo-sunset .meteo-block-data').text(sunset);

        date = date.add(1, 'days');
    }
}

function displayAlerts(data) {
    const alertsContainer = $('#weather-alerts');
    alertsContainer.empty();

    if (data.alerts && data.alerts.length > 0) {
        data.alerts.forEach(alert => {
            const alertElement = $('<div class="alert alert-warning" role="alert"></div>');
            alertElement.html(`<strong>${alert.event}</strong>: ${alert.description}`);
            alertsContainer.append(alertElement);
        });
    } else {
        const noAlertElement = $('<div class="alert alert-info" role="alert">No current weather alerts.</div>');
        alertsContainer.append(noAlertElement);
    }
    alertsContainer.show();
}

function createWeatherWidget(city) {
    const existingWidget = document.getElementById('custom-weather-widget');
    if (existingWidget) {
        existingWidget.remove();
    }

    const widgetContainer = document.createElement('div');
    widgetContainer.id = 'custom-weather-widget';
    widgetContainer.style.padding = '15px';
    widgetContainer.style.borderRadius = '10px';
    widgetContainer.style.marginTop = '15px';
    widgetContainer.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
    widgetContainer.style.fontFamily = 'Arial, sans-serif';
    widgetContainer.style.maxWidth = '300px';
    widgetContainer.style.margin = '15px auto';

    widgetContainer.innerHTML = '<p style="text-align: center; font-size: 14px;">Loading weather data for ' + city + '...</p>';

    document.querySelector('.container').appendChild(widgetContainer);

    $.ajax({
        url: `https://api.openweathermap.org/data/2.5/weather?q=${city}&units=metric&appid=${API_KEY}`,
        success: function(data) {
            updateWidgetContent(widgetContainer, data);
        },
        error: function(req, status, error) {
            widgetContainer.innerHTML = '<p style="text-align: center; font-size: 14px; color: red;">Error fetching weather data for ' + city + '. Please try again.</p>';
            console.error('Error:', error);
        }
    });
}

function updateWidgetContent(widgetContainer, data) {
    const temp = data.main.temp;
    let bgColor, textColor;

    if (temp < 0) {
        bgColor = 'linear-gradient(135deg, #7eb5ff, #4e95ff)';
        textColor = 'white';
    } else if (temp < 10) {
        bgColor = 'linear-gradient(135deg, #8fdaff, #46c1ff)';
        textColor = 'black';
    } else if (temp < 20) {
        bgColor = 'linear-gradient(135deg, #ffee82, #fad200)';
        textColor = 'black';
    } else if (temp < 30) {
        bgColor = 'linear-gradient(135deg, #ffb88e, #ff8a3d)';
        textColor = 'black';
    } else {
        bgColor = 'linear-gradient(135deg, #ff9696, #ff5858)';
        textColor = 'white';
    }

    widgetContainer.style.background = bgColor;
    widgetContainer.style.color = textColor;

    widgetContainer.innerHTML = `
        <h3 style="text-align: center; margin-bottom: 10px; font-size: 18px;">${data.name}</h3>
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center;">
                <img src="http://openweathermap.org/img/wn/${data.weather[0].icon}.png" alt="${data.weather[0].description}" style="width: 50px; height: 50px;">
                <p style="font-size: 14px; margin: 5px 0;">${data.weather[0].description}</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 24px; font-weight: bold; margin: 0;">${Math.round(data.main.temp)}°C</p>
                <p style="font-size: 14px; margin: 5px 0;">Feels like: ${Math.round(data.main.feels_like)}°C</p>
            </div>
        </div>
        <div style="display: flex; justify-content: space-around; margin-top: 10px; font-size: 12px;">
            <p><strong>Humidity:</strong> ${data.main.humidity}%</p>
            <p><strong>Wind:</strong> ${data.wind.speed} m/s</p>
        </div>
    `;
}



// ... (previous code remains the same)

function displayMeteo(data) {
    var googleMapCity = "https://www.google.fr/maps/place/" + data.city.coord.lat + "," + data.city.coord.lon;
    $('#meteo-title span').html('Weather in <a href="' + googleMapCity + '" class="text-muted meteo-city" target="_blank">' + data.city.name + ', ' + data.city.country + '</a>');

    var tempMoyenne = 0;
    displayAlerts(data);
    displaySunriseSunset(data.city.coord.lat, data.city.coord.lon);

    for (var i = 0; i < 3; i++) {
        var meteo = data.list[i * 8];
        var day = $("#meteo-day-" + (i + 1));
        var icon = day.find(".meteo-temperature .wi");
        var temperature = day.find(".meteo-temperature .data");
        var humidity = day.find(".meteo-humidity .meteo-block-data");
        var wind = day.find(".meteo-wind .meteo-block-data");

        var code = meteo.weather[0].id;
        icon.attr('class', 'wi wi-owm-' + code);
        temperature.text(toCelsius(meteo.main.temp) + "°C");
        humidity.text(meteo.main.humidity + "%");
        wind.text(meteo.wind.speed + " km/h");
        tempMoyenne += meteo.main.temp;
    }

    tempMoyenne = toCelsius(tempMoyenne / 3);
    var hue1 = 30 + 240 * (30 - tempMoyenne) / 60;
    var hue2 = hue1 + 30;
    var rgb1 = 'rgb(' + hslToRgb(hue1 / 400, 0.7, 0.5).join(',') + ')';
    var rgb2 = 'rgb(' + hslToRgb(hue2 / 400, 0.7, 0.5).join(',') + ')';
    $('body').css('background', 'linear-gradient(' + rgb1 + ',' + rgb2 + ')');

    createWeatherWidget(data.city.name);
    integrateHistoricalWeather(data.city);
    displayHistoricalWeather(historicalData);
}

function integrateHistoricalWeather(city) {
    console.log("Integrating historical weather for:", city.name);
    
    // Create or clear the historical weather section
    let historicalSection = $('#historical-weather');
    if (historicalSection.length === 0) {
        historicalSection = $('<div id="historical-weather" class="mt-4"></div>');
        $('#meteo-title').after(historicalSection);
    } else {
        historicalSection.empty();
    }
    
    historicalSection.html('<h3>Historical Weather (Past 5 Days)</h3><p>Loading historical data...</p>');

    getHistoricalWeather(city, function(data, error) {
        if (error === null && data) {
            console.log("Data received in integrateHistoricalWeather:", data);
            displayHistoricalWeather(data);
        } else {
            console.error("Error fetching historical weather data:", error);
            historicalSection.html('<h3>Historical Weather (Past 5 Days)</h3><p>Error loading historical data. Please try again later.</p>');
        }
    });
}

function getHistoricalWeather(city, callback) {
    console.log("Fetching historical weather for:", city.name);
    const endDate = Math.floor(Date.now() / 1000); // Current timestamp
    const historicalDays = 5; // Number of past days

    const historicalData = [];

    let completedRequests = 0;
    const checkAllRequestsComplete = () => {
        completedRequests++;
        if (completedRequests === historicalDays) {
            console.log("All historical data requests completed. Data:", historicalData);
            callback(historicalData, null);
        }
    };

    for (let i = 1; i <= historicalDays; i++) {
        const dayTimestamp = endDate - i * 24 * 60 * 60; // Subtract 'i' days from current date

        $.ajax({
            url: `https://api.openweathermap.org/data/3.0/onecall/timemachine?lat=${city.coord.lat}&lon=${city.coord.lon}&dt=${dayTimestamp}&appid=${API_KEY}`,
            success: function(data) {
                console.log(`Received data for day -${i}:`, data);
                historicalData.push(data);
                checkAllRequestsComplete();
            },
            error: function(req, status, error) {
                console.error(`Error fetching historical data for day -${i}:`, status, error);
                console.log("Response:", req.responseText);
                historicalData.push(null); // Push null for failed requests
                checkAllRequestsComplete();
            }
        });
    }
}

function displayHistoricalWeather(historicalData) {
    console.log("Displaying historical data:", JSON.stringify(historicalData, null, 2));

    const historicalSection = $('#historical-weather');
    historicalSection.empty();

    const header = $('<h3>Historical Weather (Past 5 Days)</h3>');
    historicalSection.append(header);

    if (!historicalData || historicalData.length === 0) {
        historicalSection.append('<p>No historical data available.</p>');
        return;
    }

    const table = $('<table class="table table-striped"></table>');
    const tableHeader = $('<thead><tr><th>Date</th><th>Temperature</th><th>Humidity</th><th>Wind Speed</th><th>Conditions</th></tr></thead>');
    const tableBody = $('<tbody></tbody>');

    table.append(tableHeader);
    table.append(tableBody);

    let dataFound = false;

    historicalData.forEach((dayData, index) => {
        if (dayData && dayData.data && dayData.data[0]) {
            dataFound = true;
            const date = new Date(dayData.data[0].dt * 1000);
            const row = $('<tr></tr>');
            row.append(`<td>${date.toLocaleDateString()}</td>`);
            row.append(`<td>${toCelsius(dayData.data[0].temp)}°C</td>`);
            row.append(`<td>${dayData.data[0].humidity}%</td>`);
            row.append(`<td>${dayData.data[0].wind_speed} m/s</td>`);
            row.append(`<td>${dayData.data[0].weather[0].description}</td>`);
            tableBody.append(row);
        } else {
            console.warn(`No valid data for day ${index + 1}:`, dayData);
        }
    });

    if (dataFound) {
        historicalSection.append(table);
    } else {
        historicalSection.append('<p>Unable to retrieve historical data. Please try again later.</p>');
    }
}


function checkAPIKey() {
    $.ajax({
        url: `https://api.openweathermap.org/data/2.5/weather?q=London&appid=${API_KEY}`,
        success: function(data) {
            console.log("API key is valid");
        },
        error: function(req, status, error) {
            console.error("API key may be invalid:", status, error);
            alert("There might be an issue with the API key. Please check the console for more information.");
        }
    });
}

// Initialize the widget on page load
$(document).ready(function() {
    var defaultCity = document.location.hash ? document.location.hash.substr(1) : "London";
    createWeatherWidget(defaultCity);
    checkAPIKey();
});