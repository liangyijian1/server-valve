#include <cstdio>
#include <cstdio>
#include <winsock2.h>
#include "_utils.h"
#include "MvCameraControl.h"
#include "cmdline.h"

#define BUF_LEN  100
#pragma comment(lib,"ws2_32.lib")

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    cmdline::parser a;
    a.add<string>("path", 'p', "tem path", true, "");
    a.parse_check(argc, argv);
    string tempath = a.get<string>("path");
    _utils u;
    u.m_matDst = imread(tempath, 0);
    cout << u.m_matDst.size() << '\n';

    WSADATA wd;
    SOCKET ServerSock, ClientSock;
    char Buf[BUF_LEN] = { 0 };
    SOCKADDR ClientAddr;
    SOCKADDR_IN ServerSockAddr;
    int addr_size = 0, recv_len = 0;
    VideoCapture capture(0);
    capture.set(CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(CAP_PROP_FRAME_WIDTH, 5472);
    capture.set(CAP_PROP_FRAME_HEIGHT, 3648);

    /* ��ʼ������sock��Ҫ��DLL */
    WSAStartup(MAKEWORD(2, 2), &wd);
    /* ���������socket */
    if (-1 == (ServerSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)))
    {
        printf("socket error!\n");
        exit(1);
    }
    /* ���÷������Ϣ */
    memset(&ServerSockAddr, 0, sizeof(ServerSockAddr));  // ���ṹ��ServerSockAddr����
    ServerSockAddr.sin_family = AF_INET;       // ʹ��IPv4��ַ
    ServerSockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");// ����IP��ַ
    ServerSockAddr.sin_port = htons(1314);      // �˿�
    /* ���׽��� */
    if (-1 == ::bind(ServerSock, (SOCKADDR*)&ServerSockAddr, sizeof(SOCKADDR)))
    {
        printf("bind error!\n");
        exit(1);
    }
    /* �������״̬ */
    if (-1 == listen(ServerSock, 10))
    {
        printf("listen error!\n");
        exit(1);
    }
    addr_size = sizeof(SOCKADDR);
    while (true)
    {
        /* �����ͻ�������accept��������һ���µ��׽��֣����ͺͽ��ն���������׽��� */
        if (-1 == (ClientSock = accept(ServerSock, (SOCKADDR*)&ClientAddr, &addr_size)))
        {
            printf("socket error!\n");
            exit(1);
        }
        /* ���ܿͻ��˵ķ������� */
        int recv_len = recv(ClientSock, Buf, BUF_LEN, 0);
        printf("�ͻ��˷��͹���������Ϊ��%s\n", Buf);
        //��������2

        Mat frame;
        capture >> frame;
        cvtColor(frame, u.m_matSrc, COLOR_BGR2GRAY);

        u.Match();
        u.m_matSrc.release();
        frame.release();
        /* �������ݵ��ͻ��� */
        send(ClientSock, Buf, recv_len, 0);
        /* �رտͻ����׽��� */
        closesocket(ClientSock);
        /* ��ջ����� */
        memset(Buf, 0, BUF_LEN);
    }

    /* �رշ�����׽��� */
    //closesocket(ServerSock);
    /* WSACleanup();*/
    return 0;
}