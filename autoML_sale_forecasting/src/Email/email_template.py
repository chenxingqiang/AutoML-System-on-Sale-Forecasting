# -*- encoding: utf-8 -*-
import sys
import time

import os
import smtplib
import socket
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader


class Mail:
    SERVER = ''
    SERVER_PORT = ''

    FROM = ''
    SENDER = ''
    PASSWD = ''

    RETRY = 10
    RETRY_INIT_DELAY = 15

    template_env = Environment(autoescape=True,
                               loader=FileSystemLoader(os.getcwd()))

    def __init__(self, subject, to, cc=None):
        self.mail = MIMEMultipart('related')

        self.mail['Subject'] = subject
        self.mail['From'] = Mail.FROM
        self.mail['To'] = ';'.join(to)
        if cc: self.mail['Cc'] = ';'.join(cc)

        self.__inject_jinja2_env()

    def __inject_jinja2_env(self):
        def percent(number):
            try:
                return '%.2f%%' % (float(number) * 100.0)
            except ValueError:
                return 'n.a.'

        def dayofweek(date):
            week_map = [u'周日', u'周一', u'周二', u'周三', u'周四', u'周五', u'周六']
            if isinstance(date, int):
                return week_map[date]
            else:
                import time
                return week_map[int(time.strftime('%w', time.strptime(date, '%Y-%m-%d')))]

        def percent_in(a, b=None):
            try:
                if float(b) == 0:
                    return 'n.a.'
                elif b is None:
                    return percent(a)
                else:
                    return '%.2f%%' % (float(a) / float(b) * 100.0)
            except:
                return 'n.a.'

        def to_utf8(o):
            return str(o).decode('utf-8')

        def seq2url(seq):
            if seq is None or '_' not in seq: return ''
            city, dt = seq.rsplit('_', 1)
            return '%s/dt-%s' % (city, dt)

        def round_wrapped(a, n=0):
            if a == 'NULL': return 0
            if n <= 0:
                return int(round(float(a)))
            else:
                return round(float(a), n)

        self.template_env.filters['percent'] = percent
        self.template_env.filters['dayofweek'] = dayofweek
        self.template_env.filters['percent_in'] = percent_in
        self.template_env.filters['to_utf8'] = to_utf8
        self.template_env.filters['seq2url'] = seq2url
        self.template_env.filters['round'] = round_wrapped

    def add_text(self, text):
        text_part = MIMEText(text, 'plain', 'utf-8')
        self.mail.attach(text_part)

    def add_rendered_html(self, template='mail.html', **data):
        template_to_render = Mail.template_env.get_template(template)
        html_part = MIMEText(template_to_render.render(**data), 'html', 'utf-8')
        self.mail.attach(html_part)

    def add_image(self, target, cid=None, in_content=False):
        if in_content:
            embedded_img_part = MIMEText('<img src="cid:%s">' % cid, 'html', 'utf-8')
            self.mail.attach(embedded_img_part)

        if isinstance(target, str):
            with open(target, 'rb') as f:
                content = f.read()
            if cid is None:
                cid = os.path.basename(target)
        elif hasattr(target, 'read'):
            content = target.read()
        else:
            raise ValueError('target should be str or file-like object')

        image_part = MIMEImage(content)
        image_part.add_header('Content-ID', '<%s>' % cid)
        image_part.add_header('Content-Description', cid)
        image_part.add_header('Content-Disposition', 'inline', filename=cid + '.jpg')
        self.mail.attach(image_part)

    def add_image_list(self, target_list, cid_list, in_content=False):
        if in_content:
            html_str = ''
            for cid in cid_list:
                html_str = html_str + '<img src="cid:%s">' % cid
            embedded_img_part = MIMEText(html_str, 'html', 'utf-8')
            self.mail.attach(embedded_img_part)

        for i, target in enumerate(target_list):
            cid = cid_list[i]
            if isinstance(target, str):
                with open(target, 'rb') as f:
                    content = f.read()
                if cid is None:
                    cid = os.path.basename(target)
            elif hasattr(target, 'read'):
                content = target.read()
            else:
                raise ValueError('target should be str or file-like object')

            image_part = MIMEImage(content)
            image_part.add_header('Content-ID', '<%s>' % cid)
            image_part.add_header('Content-Description', cid)
            image_part.add_header('Content-Disposition', 'inline', filename=cid + '.jpg')
            self.mail.attach(image_part)

    def add_attachment(self, path, filename):
        with open(path, 'rb') as f:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(f.read())
            encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", 'attachment', filename=os.path.basename(filename))
            self.mail.attach(attachment)

    def send(self, debug=False):
        print >> sys.stderr, "Sending mail to %s" % self.mail['To']
        if self.mail['Cc']: print >> sys.stderr, "cc to %s" % self.mail['Cc']

        delay = Mail.RETRY_INIT_DELAY
        last_exception = None
        content = self.mail.as_string()

        for attempt in range(Mail.RETRY_INIT_DELAY):
            s = smtplib.SMTP()
            s.set_debuglevel(debug)
            try:
                s.connect(Mail.SERVER, Mail.SERVER_PORT)
                s.starttls()
                s.login(Mail.SENDER, Mail.PASSWD)
                send_to = self.mail['To'].split(';')
                if self.mail['Cc']: send_to += self.mail['Cc'].split(';')
                ret = s.sendmail(Mail.FROM, send_to, content)

                for to_addr, err in ret.iteritems():
                    (code, resp) = err
                    print >> sys.stderr, 'Sending to %s failed, code %d, reason %s' % (to_addr, code, resp)
                s.quit()
                s.close()
            except smtplib.SMTPResponseException as e:
                print("Sending try #%d fail: %s, will retry in %s seconds" % (attempt + 1, e, delay))
                last_exception = e
                time.sleep(delay)
                delay *= 2
                if delay >= 120: delay = 120
            except smtplib.SMTPConnectError as e:
                print("Sending try #%d fail: %s, will retry in %s seconds" % (attempt + 1, e, delay))
                last_exception = e
                time.sleep(delay)
                delay *= 2
                if delay >= 120: delay = 120
            except smtplib.SMTPServerDisconnected as e:
                print("Sending try #%d fail: %s, will retry in %s seconds" % (attempt + 1, e, delay))
                last_exception = e
                time.sleep(delay)
                delay *= 2
                if delay >= 120: delay = 120
            except socket.error as e:
                print("Sending try #%d fail: %s, will retry in %s seconds" % (attempt + 1, e, delay))
                last_exception = e
                time.sleep(delay)
                delay *= 2
                if delay >= 120: delay = 120
            else:
                break
            finally:
                # s.quit()
                # s.close()
                pass
        else:
            print("Fail after %d attempts" % Mail.RETRY)
            raise last_exception

    def template_filter(self, name, func):
        self.template_env.filters[name] = func

    def __str__(self):
        print(self.mail.as_string())
