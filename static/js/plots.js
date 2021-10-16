$('#first_cat').on('change',function(){

    $.ajax({
        url: "/bar",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('first_cat').value
        },
        dataType:"json",
        success: function (data) {
            Plotly.newPlot('chart', data[0] );
            Plotly.newPlot('chart2', data[1], {});
            Plotly.newPlot('chart3', data[2], {});
        }
    });
})
