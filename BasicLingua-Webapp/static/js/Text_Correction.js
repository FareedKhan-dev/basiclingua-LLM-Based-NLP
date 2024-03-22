window.onload = function() {
    var cachedApiKey = localStorage.getItem("api_key");
    var cachedUserInput = localStorage.getItem("user_input");
    

    
    document.getElementById("api_key").value = cachedApiKey || "";
    document.getElementById("user_input").value = cachedUserInput && "";
    
    
};

    document.getElementById("process-button").addEventListener("click", function() {
        var apiKey = document.getElementById("api_key").value.trim();
        var userInput = document.getElementById("user_input").value.trim();

        if (apiKey === "" || userInput === "") {
            alert("Please fill in all required fields.");
            return;
        }

        document.getElementById("loader").style.display = "block";

        localStorage.setItem("api_key", apiKey);
        localStorage.setItem("user_input", userInput);

        $.ajax({
            url: translationUrl,
            type: "POST",
            data: $("#translation-form").serialize(),
            success: function(response) {
                document.getElementById("loader").style.display = "none";
                console.log("Response:", response);
                document.getElementById("corrected-result").innerHTML = "<h4>Corrected Text</h4><p style='text-align: left;'>" + response.corrected_text + "</p>";
            },
            error: function(xhr, errmsg, err) {
                document.getElementById("loader").style.display = "none";
                document.getElementById("corrected-result").innerHTML = "<h4>Error retrieving output</h4><p style='text-align: left;'>" + xhr.status + "</p>";

                console.log("Error:", xhr.status + ": " + xhr.responseText);
            }
        });
    });


        document.getElementById("refresh-button").addEventListener("click", function() {
            document.getElementById("user_input").value = "";
            document.getElementById("corrected-result").innerHTML = "";

            
        localStorage.removeItem("user_input");
        localStorage.removeItem("target_lang");
    });