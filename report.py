

def mainReportHTML(summary_table_1,summary_table_2):
    html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
    <body>
        <h1></h1>

        <!-- *** Section 1 *** --->
        <h2>train-ads-anomaly-dest-country</h2>
        <iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
src="''' + 'first_plot_url +' '''.embed?width=800&height=550"></iframe>
        <p>Apple stock price rose steadily through 2014.</p>
        
        <!-- *** Section 2 *** --->
        <h2>Section 2: AAPL compared to other 2014 stocks</h2>
        <iframe width="1000" height="1000" frameborder="0" seamless="seamless" scrolling="no" \
src="''' + 'second_plot_url' + '''.embed?width=1000&height=1000"></iframe>
        <p>aaa</p>
        <h3>bbbb</h3>
        ''' + summary_table_2 + '''
        <h3>cccc</h3>
        ''' + summary_table_1 + '''
    </body>
</html>'''

    return html_string