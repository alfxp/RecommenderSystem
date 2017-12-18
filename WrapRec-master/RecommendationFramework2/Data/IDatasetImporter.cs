﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RF2.Data
{
    public interface IDatasetImporter
    {
        void ImportData(RecSysContext recSysContext, Dataset datasetRecord);
        string GetDatasetName();
    }
}
