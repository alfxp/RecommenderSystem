﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RF2.Utilities
{
    public class Paths
    {
        // MovieLens
        public static string MovieLens1M = @"D:\Data\Datasets\MovieLens\ml-1m\ratings.dat";
        public static string MovieLens1MTrain75 = @"D:\Data\Datasets\MovieLens\ml-1m\ratings75.dat";
        public static string MovieLens1MTest25 = @"D:\Data\Datasets\MovieLens\ml-1m\ratings25.dat";
        public static string MovieLens1MUsersCluster = @"D:\Data\Datasets\MovieLens\ml-1m\Clusters\UsersCluster";
        public static string MovieLens1MItemsCluster = @"D:\Data\Datasets\MovieLens\ml-1m\Clusters\ItemsCluster";
        public static string MovieLens1MUsersRatingsCount = @"D:\Data\Datasets\MovieLens\ml-1m\UsersRatingsCount.csv";
        public static string MovieLens1MItemsRatingsCount = @"D:\Data\Datasets\MovieLens\ml-1m\ItemsRatingsCount.csv";

        // Epinion
        public static string EpinionRatings = @"D:\Data\Datasets\Epinion\ratings_data.txt";
        public static string EpinionTrain75 = @"D:\Data\Datasets\Epinion\ratings_data_75.txt";
        public static string EpinionTest25 = @"D:\Data\Datasets\Epinion\ratings_data_25.txt";
        public static string EpinionItemsCluster = @"D:\Data\Datasets\Epinion\Clusters\ItemsCluster";
        public static string EpinionUsersCluster = @"D:\Data\Datasets\Epinion\Clusters\UsersCluster";

        public static string EpinionTrain80 = @"D:\Data\Dropbox\TUDelft joint paper\data\Epinion\ratings_train_80.txt";
        public static string EpinionTest20 = @"D:\Data\Dropbox\TUDelft joint paper\data\Epinion\ratings_test_20.txt";
        public static string EpinionTrain50 = @"D:\Data\Dropbox\TUDelft joint paper\data\Epinion\ratings_train_50.txt";
        public static string EpinionTest50 = @"D:\Data\Dropbox\TUDelft joint paper\data\Epinion\ratings_test_50.txt";
        public static string EpinionRelations = @"D:\Data\Dropbox\TUDelft joint paper\data\Epinion\trust_data.txt";
        public static string EpinionRelationsImplicit = @"D:\Data\Dropbox\TUDelft joint paper\data\trust scores\";

        //Amazon
        public static string AmazonBooksRatings = @"D:\Data\Datasets\Amazon\Selected\books_selected.csv";
        public static string AmazonMusicRatings = @"D:\Data\Datasets\Amazon\Selected\music_selected.csv";
        public static string AmazonDvdRatings = @"D:\Data\Datasets\Amazon\Selected\dvd_selected.csv";
        public static string AmazonVideoRatings = @"D:\Data\Datasets\Amazon\Selected\video_selected.csv";

        public static string AmazonBooksUsersCluster = @"D:\Data\Datasets\Amazon\Selected\BookClusters\UsersCluster";
        public static string AmazonBooksItemsCluster = @"D:\Data\Datasets\Amazon\Selected\BookClusters\ItemsCluster";

        public static string AmazonMusicUsersCluster = @"D:\Data\Datasets\Amazon\Selected\MusicClusters\UsersCluster";
        public static string AmazonMusicItemsCluster = @"D:\Data\Datasets\Amazon\Selected\MusicClusters\ItemsCluster";

        public static string AmazonVideoUsersCluster = @"D:\Data\Datasets\Amazon\Selected\VideoClusters\UsersCluster";
        public static string AmazonVideoItemsCluster = @"D:\Data\Datasets\Amazon\Selected\VideoClusters\ItemsCluster";

        public static string AmazonDvdUsersCluster = @"D:\Data\Datasets\Amazon\Selected\DvdClusters\UsersCluster";
        public static string AmazonDvdItemsCluster = @"D:\Data\Datasets\Amazon\Selected\DvdClusters\ItemsCluster";

        public static string AmazonBooksTrain75 = @"D:\Data\Datasets\Amazon\Selected\books_selected_75.csv";
        public static string AmazonBooksTest25 = @"D:\Data\Datasets\Amazon\Selected\books_selected_25.csv";

        public static string AmazonMusicTrain75 = @"D:\Data\Datasets\Amazon\Selected\music_selected_75.csv";
        public static string AmazonMusicTest25 = @"D:\Data\Datasets\Amazon\Selected\music_selected_25.csv";

    }
}
