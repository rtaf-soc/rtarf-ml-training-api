

def mainReportHTML(title,summary_table):
    html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
        <style>
            .right-aligned {
                text-align: right;
                padding-right: 10px; /* เพิ่มระยะห่างด้านขวาเพื่อให้มีพื้นที่ว่างก่อนเครื่องหมายจุล */
            }
        </style>
    </head>
    <body>
        <h1>Report</h1>
        <h2>''' + title + '''</h2>
        ''' + summary_table + '''
    </body>
</html>'''

    return html_string